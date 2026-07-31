"""Opportunity-aware routing over frozen residual portfolio continuations.

The existing portfolio-race gate estimates which precommitted completion best
matches the observed pilot lanes.  This decorator adds a separate question:
did the pilot expose enough absolute opportunity to keep spending on the
current market at all?

All candidate branches still exist before current outcomes are observed.  A
weak pilot can therefore hand off only to a pre-generated renewal branch; it
cannot trigger an unlogged provider call or outcome-conditioned candidate
construction.  Later campaign stages may use the resulting authenticated
evidence to open a fresh proposal market through the ordinary expert ports.

The policy is workload-, objective-, model-, provider-, prompt-, and
configuration-blind.  Its absolute gain reference must be bound to an
authenticated prior-evidence digest so that a caller cannot silently tune the
threshold after seeing the current pilot.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.precommitted_portfolio_racing import (
    EvidenceAdaptivePortfolioRaceGate,
    PortfolioRaceGateDecision,
    PrecommittedPortfolioRacePlan,
)
from agent_evolve.application.sequential_lineage_allocation import (
    SequentialAllocationGateDecisionPort,
    SequentialAllocationPlanPort,
    SequentialPilotOutcomeBatch,
)
from agent_evolve.domain.patch import require_sha256
from agent_evolve.domain.typed_json import freeze_json, thaw_json


CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_ID = (
    "causal_opportunity_portfolio_race_gate"
)
CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_VERSION = 2

_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = (
    b"agent-evolve:causal-opportunity-portfolio-race-gate:v2\x00"
)


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
        raise ValueError(f"{name} must use the closed token grammar")


def _resolve_frozen_route(
    plan: PrecommittedPortfolioRacePlan,
    requested_branch_id: str,
) -> tuple[str, tuple[str, ...]]:
    """Resolve a configured source branch to its frozen equivalent.

    A disagreement planner may outcome-blindly collapse source branches that
    produce the same complete action slate, either before or after the common
    pilot is inserted.  The planner authenticates that exact equivalence in
    ``outcome_blind_route_resolution``.  The gate may consume that proof, but
    it must never guess a substitute for a non-equivalent missing route.
    """

    _require_token(requested_branch_id, name="requested_branch_id")
    if requested_branch_id in plan.frozen_branch_ids:
        return requested_branch_id, ("identity",)
    raw_plan_evidence = thaw_json(plan.evidence)
    if type(raw_plan_evidence) is not dict:
        raise TypeError("portfolio plan evidence must have an object root")
    raw_design = raw_plan_evidence.get("disagreement_design")
    if type(raw_design) is not dict:
        raise ValueError(
            "configured route is absent and the plan has no disagreement proof"
        )
    raw_design_evidence = raw_design.get("evidence")
    if type(raw_design_evidence) is not dict:
        raise ValueError(
            "configured route is absent and disagreement evidence is missing"
        )
    raw_rows = raw_design_evidence.get(
        "outcome_blind_route_resolution"
    )
    if type(raw_rows) is not list:
        raise ValueError(
            "configured route is absent and has no authenticated resolution"
        )
    matches = tuple(
        value
        for value in raw_rows
        if type(value) is dict
        and value.get("source_branch_id") == requested_branch_id
    )
    if len(matches) != 1:
        raise ValueError(
            "configured route lacks one unambiguous authenticated resolution"
        )
    row = matches[0]
    representative = row.get("representative_branch_id")
    raw_path = row.get("equivalence_path")
    if type(representative) is not str:
        raise TypeError("resolved representative_branch_id must be a string")
    _require_token(representative, name="representative_branch_id")
    if (
        type(raw_path) is not list
        or not raw_path
        or any(type(value) is not str for value in raw_path)
    ):
        raise ValueError("route resolution needs an equivalence proof path")
    path = tuple(raw_path)
    allowed = {
        "identity",
        "identical_complete_action_set_before_pilot",
        "identical_complete_action_set_after_frozen_pilot",
    }
    if any(value not in allowed for value in path):
        raise ValueError("route resolution uses an unknown equivalence proof")
    if representative not in plan.frozen_branch_ids:
        raise ValueError(
            "authenticated route representative is absent from the frozen plan"
        )
    return representative, path


@dataclass(frozen=True, slots=True)
class CausalOpportunityPortfolioRaceGate:
    """Route weak, sparse, and productive pilots to frozen continuations.

    ``reference_gain_scale`` is a robust positive-gain scale derived only from
    strictly prior authenticated outcomes.  ``minimum_peak_gain_ratio`` tests
    whether the best pilot clears a declared fraction of that scale.
    ``minimum_positive_fraction`` separates a sparse market from a productive
    one after the absolute opportunity check passes.

    The productive route delegates to the existing lane-aware race gate.  The
    weak and sparse routes select explicit precommitted branch IDs.  This
    separation is intentional: opportunity detection and within-market
    conversion are different control problems.
    """

    renewal_branch_id: str
    exploration_branch_id: str
    reference_gain_scale: float
    reference_gain_evidence_sha256: str
    minimum_peak_gain_ratio: float = 0.5
    minimum_positive_fraction: float = 0.5
    minimum_pilot_count: int = 2
    base_gate: EvidenceAdaptivePortfolioRaceGate = field(
        default_factory=EvidenceAdaptivePortfolioRaceGate,
        repr=False,
        compare=False,
    )
    gate_id: str = CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_ID
    gate_version: int = CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_VERSION
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_token(self.renewal_branch_id, name="renewal_branch_id")
        _require_token(
            self.exploration_branch_id,
            name="exploration_branch_id",
        )
        if self.renewal_branch_id == self.exploration_branch_id:
            raise ValueError(
                "renewal and exploration branches must be distinct"
            )
        if (
            type(self.reference_gain_scale) is not float
            or not math.isfinite(self.reference_gain_scale)
            or self.reference_gain_scale <= 0.0
        ):
            raise ValueError(
                "reference_gain_scale must be finite and positive"
            )
        require_sha256(
            self.reference_gain_evidence_sha256,
            "reference_gain_evidence_sha256",
        )
        if (
            type(self.minimum_peak_gain_ratio) is not float
            or not math.isfinite(self.minimum_peak_gain_ratio)
            or self.minimum_peak_gain_ratio < 0.0
        ):
            raise ValueError(
                "minimum_peak_gain_ratio must be finite and non-negative"
            )
        if (
            type(self.minimum_positive_fraction) is not float
            or not math.isfinite(self.minimum_positive_fraction)
            or not 0.0 <= self.minimum_positive_fraction <= 1.0
        ):
            raise ValueError(
                "minimum_positive_fraction must lie in [0, 1]"
            )
        if (
            type(self.minimum_pilot_count) is not int
            or self.minimum_pilot_count <= 0
        ):
            raise ValueError("minimum_pilot_count must be positive")
        if type(self.base_gate) is not EvidenceAdaptivePortfolioRaceGate:
            raise TypeError("base_gate must be an exact evidence-adaptive gate")
        self.base_gate.__post_init__()
        _require_token(self.gate_id, name="gate_id")
        if self.gate_version != CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_VERSION:
            raise ValueError("gate_version is immutable")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                _DEFINITION_DOMAIN,
                {
                    "schema_version": 1,
                    "gate_id": self.gate_id,
                    "gate_version": self.gate_version,
                    "renewal_branch_id": self.renewal_branch_id,
                    "exploration_branch_id": self.exploration_branch_id,
                    "reference_gain_scale_hex": (
                        self.reference_gain_scale.hex()
                    ),
                    "reference_gain_evidence_sha256": (
                        self.reference_gain_evidence_sha256
                    ),
                    "minimum_peak_gain_ratio_hex": (
                        self.minimum_peak_gain_ratio.hex()
                    ),
                    "minimum_positive_fraction_hex": (
                        self.minimum_positive_fraction.hex()
                    ),
                    "minimum_pilot_count": self.minimum_pilot_count,
                    "base_gate": {
                        "gate_id": self.base_gate.gate_id,
                        "gate_version": self.base_gate.gate_version,
                        "definition_sha256": (
                            self.base_gate.definition_sha256
                        ),
                    },
                    "routing_order": (
                        "insufficient_pilot_then_weak_absolute_opportunity_"
                        "then_sparse_positive_support_then_lane_adaptation"
                    ),
                    "missing_route_resolution": (
                        "authenticated_exact_action_set_equivalence_only"
                    ),
                    "candidate_branches_frozen_before_outcomes": True,
                    "outcome_conditioned_regeneration": False,
                    "only_pilot_outcomes_observed": True,
                    "workload_model_provider_branches": False,
                },
            ),
        )

    def decide(
        self,
        plan: SequentialAllocationPlanPort,
        outcomes: SequentialPilotOutcomeBatch,
    ) -> SequentialAllocationGateDecisionPort:
        self.__post_init__()
        if type(plan) is not PrecommittedPortfolioRacePlan:
            raise TypeError(
                "causal opportunity gate requires an exact portfolio plan"
            )
        plan.__post_init__()
        if type(outcomes) is not SequentialPilotOutcomeBatch:
            raise TypeError("outcomes must be exact")
        outcomes.__post_init__()

        base_decision = self.base_gate.decide(plan, outcomes)
        if type(base_decision) is not PortfolioRaceGateDecision:
            raise TypeError("base gate returned a foreign decision")
        base_decision.__post_init__()

        pilot_count = len(outcomes.outcomes)
        positive_count = sum(
            value.positive_marginal_utility
            for value in outcomes.outcomes
        )
        maximum_gain = max(
            (
                value.marginal_archive_gain
                for value in outcomes.outcomes
            ),
            default=0.0,
        )
        peak_gain_ratio = maximum_gain / self.reference_gain_scale
        positive_fraction = (
            0.0 if pilot_count == 0 else positive_count / pilot_count
        )

        if pilot_count < self.minimum_pilot_count:
            requested_branch_id = base_decision.selected_branch_id
            route = "insufficient_pilot_base_gate"
        elif (
            positive_count == 0
            or peak_gain_ratio < self.minimum_peak_gain_ratio
        ):
            requested_branch_id = self.renewal_branch_id
            route = "weak_opportunity_renewal"
        elif positive_fraction < self.minimum_positive_fraction:
            requested_branch_id = self.exploration_branch_id
            route = "sparse_opportunity_exploration"
        else:
            requested_branch_id = base_decision.selected_branch_id
            route = "productive_market_lane_adaptation"

        selected_branch_id, route_equivalence_path = _resolve_frozen_route(
            plan,
            requested_branch_id,
        )
        selected = plan.branch_for(selected_branch_id)
        return PortfolioRaceGateDecision(
            gate_id=self.gate_id,
            gate_version=self.gate_version,
            gate_definition_sha256=self.definition_sha256,
            plan_sha256=plan.plan_sha256,
            pilot_outcome_batch_sha256=outcomes.batch_sha256,
            selected_branch_id=selected_branch_id,
            selected_requirement_sha256=(
                selected.requirement.requirement_sha256
            ),
            positive_pilot_count=positive_count,
            pilot_count=pilot_count,
            branch_scores=base_decision.branch_scores,
            evidence=freeze_json(
                {
                    "route": route,
                    "requested_branch_id": requested_branch_id,
                    "resolved_branch_id": selected_branch_id,
                    "route_equivalence_path": list(
                        route_equivalence_path
                    ),
                    "maximum_pilot_marginal_gain_hex": maximum_gain.hex(),
                    "reference_gain_scale_hex": (
                        self.reference_gain_scale.hex()
                    ),
                    "reference_gain_evidence_sha256": (
                        self.reference_gain_evidence_sha256
                    ),
                    "peak_gain_ratio_hex": peak_gain_ratio.hex(),
                    "minimum_peak_gain_ratio_hex": (
                        self.minimum_peak_gain_ratio.hex()
                    ),
                    "positive_pilot_count": positive_count,
                    "pilot_count": pilot_count,
                    "positive_fraction_hex": positive_fraction.hex(),
                    "minimum_positive_fraction_hex": (
                        self.minimum_positive_fraction.hex()
                    ),
                    "minimum_pilot_count": self.minimum_pilot_count,
                    "base_gate_decision": base_decision.to_record(
                        include_evidence=True
                    ),
                    "selected_branch_frozen_before_pilot": True,
                    "outcome_conditioned_regeneration": False,
                    "only_pilot_outcomes_observed": True,
                    "workload_model_provider_branches": False,
                }
            ),
        )


__all__ = [
    "CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_ID",
    "CAUSAL_OPPORTUNITY_PORTFOLIO_GATE_VERSION",
    "CausalOpportunityPortfolioRaceGate",
]

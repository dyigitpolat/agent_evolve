"""Front-proximity parent basis: stop generating from dominated parents.

The allocator-side companion of this rule
(:mod:`agent_evolve.application.front_proximity_admission`) restricts which
members of an already-generated market an allocator may buy.  This module
applies the *same* geometric quantity one level earlier, to the set of
parents a proposal market is generated FROM.

The measured motivation is in
``research_artifacts/jul28_large_gain_anatomy_and_refinement.md``: across the
eight sealed BOiLS held-out markets, 26% of every market descended from
parents whose Chebyshev excess over the archive front exceeded 0.05, and that
quarter held 1 of 88 prefix-positive members and 1.6% of all realised gain.
No allocator can recover a seat spent inside a region that carries no outcome
mass; the binding constraint is at least as much in the proposal universe as
in the allocation.  This module removes that region at its source.

The rule is a *support restriction over the parent population*, composed on
top of the unchanged
:func:`agent_evolve.application.residual_reachability.select_residual_reachability_basis`.
Every existing admission route (quality archive, initial design, earned
lineage, structural cover, capacity fill) still runs and still stamps its own
reason; the only change is which parents each route is allowed to see.

Three properties are deliberate:

* **Relative, not absolute.**  ``proximity_concentration`` names a fraction of
  the parent population, never a distance threshold, so the rule carries no
  workload constant, no metric name, no scale assumption and no objective
  count.  Anchor excess is computed from the archive front and a parent's own
  objective vector, so it is domain-agnostic by construction.
* **Floored, not banned.**  ``far_front_floor`` reserves a strictly positive
  share of the basis for parents *outside* the proximal head.  The archive
  front moves as the campaign buys points, so today's far parent can be
  tomorrow's near one; a rule that banned the distal tail outright could never
  discover that.  ``minimum_far_front_parents`` keeps the reservation
  non-empty even for a small basis.
* **A no-op without anchors.**  Parents with no recorded objective vector are
  scored at the population median, exactly as the admission rule scores
  anchorless candidates, so a lane is never systematically evicted.  If no
  parent carries an anchor at all, the selection is byte-identical to the
  unrestricted basis.

Claim boundary: this changes which parents a generator proposes from.  It
makes no claim about the outcome of any campaign that uses it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    ObjectivePoint,
)
from agent_evolve.application.front_proximity_admission import (
    chebyshev_excess,
)
from agent_evolve.application.residual_reachability import (
    ReachabilityCandidate,
    ResidualReachabilityBasis,
    ResidualReachabilityBasisPolicy,
    select_residual_reachability_basis,
)
from agent_evolve.domain.ids import CandidateId

FRONT_PROXIMITY_PARENT_BASIS_ID = "front_proximity_parent_basis"
FRONT_PROXIMITY_PARENT_BASIS_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = b"agent-evolve:front-proximity-parent-basis:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash(value: object) -> str:
    return hashlib.sha256(
        _DEFINITION_DOMAIN + _canonical_json(value)
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FrontProximityParentBasisConfig:
    """How hard the parent population is concentrated toward the front."""

    #: Fraction of the anchored parent population treated as front-proximal.
    #: 1.0 disables the restriction entirely and reproduces the unrestricted
    #: basis byte for byte.
    proximity_concentration: float = 0.50
    #: Fraction of the basis reserved for parents OUTSIDE the proximal head,
    #: so the distal tail is floored rather than banned.  0.0 is rejected:
    #: banning is not a configuration of this rule.  The floor is small by
    #: default because its purpose is to keep the distal region reachable
    #: as the front moves, not to guarantee it a large share of the basis.
    far_front_floor: float = 0.125
    #: Absolute floor on reserved distal slots, applied when the fractional
    #: reservation rounds to zero on a small basis.
    minimum_far_front_parents: int = 1

    def __post_init__(self) -> None:
        for name in ("proximity_concentration", "far_front_floor"):
            value = getattr(self, name)
            if (
                type(value) is not float
                or not math.isfinite(value)
                or not 0.0 < value <= 1.0
            ):
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.far_front_floor >= 1.0:
            raise ValueError("far_front_floor must leave a proximal share")
        if (
            type(self.minimum_far_front_parents) is not int
            or self.minimum_far_front_parents < 1
        ):
            raise ValueError(
                "minimum_far_front_parents must be a positive integer; "
                "the distal tail is floored, never banned"
            )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "proximity_concentration_hex": float(
                self.proximity_concentration
            ).hex(),
            "far_front_floor_hex": float(self.far_front_floor).hex(),
            "minimum_far_front_parents": self.minimum_far_front_parents,
        }


@dataclass(frozen=True, slots=True)
class ParentAnchorExcess:
    """One parent's Chebyshev excess over the parent population's front."""

    candidate_id: CandidateId
    excess: float
    anchored: bool

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id.value,
            "excess_hex": float(self.excess).hex(),
            "anchored": self.anchored,
        }


def parent_anchor_excesses(
    candidates: tuple[ReachabilityCandidate, ...],
    anchors: dict[CandidateId, ObjectivePoint],
    metric_ids: tuple[str, ...],
) -> tuple[ParentAnchorExcess, ...]:
    """Excess of every parent over the front of the anchored population.

    The front is the Pareto filter of every supplied anchor, so the quantity
    is relative to the archive the campaign has actually evaluated.  A parent
    on the front scores exactly 0.0; a dominated parent scores the worst-axis
    improvement it would need to reach the nearest front point.  Parents with
    no anchor take the population median, so they are neither systematically
    admitted nor systematically evicted.
    """

    if type(candidates) is not tuple or not candidates:
        raise ValueError("candidates must be a non-empty exact tuple")
    if type(anchors) is not dict:
        raise TypeError("anchors must be an exact mapping")
    if type(metric_ids) is not tuple or not metric_ids:
        raise ValueError("metric_ids must be a non-empty exact tuple")
    if tuple(sorted(metric_ids)) != metric_ids:
        raise ValueError("metric_ids must be canonically ordered")
    points = tuple(
        anchors[value.candidate_id]
        for value in candidates
        if value.candidate_id in anchors
    )
    if not points:
        return tuple(
            ParentAnchorExcess(
                candidate_id=value.candidate_id,
                excess=0.0,
                anchored=False,
            )
            for value in candidates
        )
    known: dict[CandidateId, float] = {
        value.candidate_id: chebyshev_excess(
            anchors[value.candidate_id], points, metric_ids
        )
        for value in candidates
        if value.candidate_id in anchors
    }
    ordered = sorted(known.values())
    median = ordered[len(ordered) // 2]
    return tuple(
        ParentAnchorExcess(
            candidate_id=value.candidate_id,
            excess=known.get(value.candidate_id, median),
            anchored=value.candidate_id in known,
        )
        for value in candidates
    )


@dataclass(frozen=True, slots=True)
class FrontProximityParentBasis:
    """Select a reachability basis biased toward front-proximal parents.

    ``inner_policy`` is the unmodified bounded dual-archive retention policy.
    This wrapper runs it twice — once over the proximal head with the
    unreserved slots, once over the distal remainder with the reserved floor
    — and merges the two bases.  Every admission reason is produced by the
    inner policy; none is invented here.
    """

    inner_policy: ResidualReachabilityBasisPolicy
    config: FrontProximityParentBasisConfig = field(
        default_factory=FrontProximityParentBasisConfig
    )
    policy_id: str = "front_proximity_parent_basis"
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.inner_policy) is not ResidualReachabilityBasisPolicy:
            raise TypeError(
                "inner_policy must be an exact "
                "ResidualReachabilityBasisPolicy"
            )
        self.inner_policy.__post_init__()
        if type(self.config) is not FrontProximityParentBasisConfig:
            raise TypeError("config must be an exact config")
        self.config.__post_init__()
        if _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                {
                    "arm": FRONT_PROXIMITY_PARENT_BASIS_ID,
                    "version": FRONT_PROXIMITY_PARENT_BASIS_VERSION,
                    "config": self.config.to_record(),
                    "inner_policy_definition_sha256": (
                        self.inner_policy.definition_sha256
                    ),
                    "rule": (
                        "partition the parent population by chebyshev "
                        "excess of its objective vector over the front of "
                        "that population, run the unchanged bounded "
                        "dual-archive retention over the proximal head "
                        "for the unreserved slots and over the distal "
                        "remainder for the floored slots, then merge; "
                        "unanchored parents take the population median"
                    ),
                    "outcomes_consulted": [
                        "evaluated_parent_objective_vector",
                        "quality_archive_membership",
                        "earned_positive_lineage",
                    ],
                    "forbidden_fields": [
                        "workload_id",
                        "model_id",
                        "provider_id",
                        "metric_id",
                    ],
                }
            ),
        )

    def _split_slots(self, maximum_parents: int) -> tuple[int, int]:
        """Unreserved proximal slots and the floored distal reservation."""

        reserved = max(
            self.config.minimum_far_front_parents,
            int(math.floor(self.config.far_front_floor * maximum_parents)),
        )
        reserved = min(reserved, maximum_parents - 1)
        return maximum_parents - reserved, reserved

    def proximity_order(
        self,
        excesses: tuple[ParentAnchorExcess, ...],
    ) -> tuple[CandidateId, ...]:
        """Parents ordered nearest-front first.

        Ties break on the candidate id so the order is deterministic and
        cannot depend on the order candidates were supplied in.
        """

        return tuple(
            value.candidate_id
            for value in sorted(
                excesses,
                key=lambda value: (value.excess, value.candidate_id.value),
            )
        )

    def partition(
        self,
        excesses: tuple[ParentAnchorExcess, ...],
    ) -> tuple[tuple[CandidateId, ...], tuple[CandidateId, ...]]:
        """Proximal head and distal remainder, best excess first."""

        order = self.proximity_order(excesses)
        head_size = max(
            1,
            min(
                len(order),
                int(
                    math.ceil(
                        self.config.proximity_concentration * len(order)
                    )
                ),
            ),
        )
        return order[:head_size], order[head_size:]

    def select(
        self,
        candidates: tuple[ReachabilityCandidate, ...],
        anchors: dict[CandidateId, ObjectivePoint],
        metric_ids: tuple[str, ...],
    ) -> ResidualReachabilityBasis:
        """The front-proximal basis over ``candidates``.

        Reduces to :func:`select_residual_reachability_basis` exactly when the
        concentration is 1.0 or when no parent carries an anchor.
        """

        if type(candidates) is not tuple or not candidates:
            raise ValueError("candidates must be a non-empty exact tuple")
        excesses = parent_anchor_excesses(candidates, anchors, metric_ids)
        if self.config.proximity_concentration >= 1.0 or not any(
            value.anchored for value in excesses
        ):
            return select_residual_reachability_basis(
                candidates, self.inner_policy
            )
        by_id = {value.candidate_id: value for value in candidates}
        proximal_ids, distal_ids = self.partition(excesses)
        proximal_slots, distal_slots = self._split_slots(
            self.inner_policy.maximum_parents
        )
        # Slots the proximal head cannot use spill OUTWARD in proximity
        # order rather than back to the unrestricted population, so a
        # tighter concentration can never move the basis away from the
        # front.  The reserved floor is measured against the ORIGINAL
        # partition, so growing the head never eats the reservation.
        head = self.proximity_order(excesses)[
            : max(len(proximal_ids), proximal_slots)
        ]
        members = dict(self._sub_basis(by_id, head, proximal_slots))
        for candidate_id, member in self._sub_basis(
            by_id, distal_ids, distal_slots
        ):
            members.setdefault(candidate_id, member)
        if len(members) < self.inner_policy.maximum_parents:
            # Either side of the partition can be smaller than its slot
            # allowance.  Top up from the UNRESTRICTED basis so the
            # restriction can never yield fewer parents than the policy
            # would have retained on its own.
            for candidate_id, member in self._sub_basis(
                by_id,
                tuple(sorted(by_id, key=lambda value: value.value)),
                self.inner_policy.maximum_parents,
            ):
                if len(members) >= self.inner_policy.maximum_parents:
                    break
                members.setdefault(candidate_id, member)
        return ResidualReachabilityBasis(
            policy_definition_sha256=self.inner_policy.definition_sha256,
            source_candidate_count=len(candidates),
            members=tuple(
                members[key] for key in sorted(members, key=lambda v: v.value)
            ),
        )

    def _sub_basis(
        self,
        by_id: dict[CandidateId, ReachabilityCandidate],
        candidate_ids: tuple[CandidateId, ...],
        slots: int,
    ):
        """Run the unchanged inner policy over one side of the partition."""

        if not candidate_ids or slots <= 0:
            return ()
        capped = min(slots, len(candidate_ids))
        policy = ResidualReachabilityBasisPolicy(
            maximum_parents=capped,
            maximum_quality_archive_parents=min(
                self.inner_policy.maximum_quality_archive_parents, capped
            ),
            maximum_initial_design_parents=min(
                self.inner_policy.maximum_initial_design_parents, capped
            ),
            maximum_earned_lineage_parents=min(
                self.inner_policy.maximum_earned_lineage_parents, capped
            ),
            maximum_structural_cover_parents=min(
                self.inner_policy.maximum_structural_cover_parents, capped
            ),
            policy_id=self.inner_policy.policy_id,
            policy_version=self.inner_policy.policy_version,
        )
        basis = select_residual_reachability_basis(
            tuple(by_id[value] for value in candidate_ids), policy
        )
        return tuple(
            (member.candidate.candidate_id, member)
            for member in basis.members
        )

    def evidence_record(
        self,
        candidates: tuple[ReachabilityCandidate, ...],
        anchors: dict[CandidateId, ObjectivePoint],
        metric_ids: tuple[str, ...],
    ) -> dict[str, object]:
        """Outcome-blind record of what the restriction did, for receipts."""

        excesses = parent_anchor_excesses(candidates, anchors, metric_ids)
        proximal_ids, distal_ids = self.partition(excesses)
        proximal_slots, distal_slots = self._split_slots(
            self.inner_policy.maximum_parents
        )
        anchored = tuple(value for value in excesses if value.anchored)
        return {
            "policy_id": self.policy_id,
            "version": FRONT_PROXIMITY_PARENT_BASIS_VERSION,
            "definition_sha256": self.definition_sha256,
            "config": self.config.to_record(),
            "source_parent_count": len(candidates),
            "anchored_parent_count": len(anchored),
            "proximal_pool_size": len(proximal_ids),
            "distal_pool_size": len(distal_ids),
            "unreserved_proximal_slots": proximal_slots,
            "reserved_distal_slots": distal_slots,
            "excesses": [value.to_record() for value in excesses],
        }


__all__ = [
    "FRONT_PROXIMITY_PARENT_BASIS_ID",
    "FRONT_PROXIMITY_PARENT_BASIS_VERSION",
    "FrontProximityParentBasis",
    "FrontProximityParentBasisConfig",
    "ParentAnchorExcess",
    "parent_anchor_excesses",
]

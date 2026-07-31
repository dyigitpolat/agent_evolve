"""Front-proximity admission: drop the market's dominated-parent tail.

A candidate's *parent anchor* is the objective point its action was derived
from.  The anchor's Chebyshev excess over the current archive front measures
how deep inside the dominated interior that anchor sits: a value at or below
zero means the anchor is itself non-dominated, and larger values mean the
anchor is buried further behind the front.

On the jul28 sealed BOiLS held-out markets this quantity is the strongest
outcome-blind discriminator measured (pooled AUC 0.732 for individual
positivity, 8 of 8 markets at or above 0.50), and its worst band is
effectively empty: anchors more than 0.05 normalized units behind the front
hold 26% of the market mass, 1 of 88 positives and 1.6% of the realised
gain.  This module turns that observation into a *support restriction* that
composes over any allocation policy: the wrapped policy still makes every
decision, it simply never sees the tail of the market that carries no
outcome mass.

The rule is deliberately RELATIVE — it keeps the market's own best
``keep_fraction`` of anchors — so it introduces no absolute threshold, no
workload constant, no metric name and no scale assumption.  Candidates whose
anchor is unknown are scored at the market's median anchor so an anchorless
lane is never systematically evicted or systematically protected.

The signal replicates: pooled AUC 0.700 on the five jul27 replay-corpus
markets that record parent anchors (0.583 to 0.767 per market), against
0.732 on the eight held-out markets.  What has NOT been shown is that any
deterministic policy built on it beats the ``lane_heads`` incumbent on four
sealed cells; the incumbent's own headline sits at the p = 0.08 tail of the
uniform null, so a four-cell single-run comparison cannot resolve it.

Claim boundary: replay-evaluation only.  Nothing here carries live campaign
authority.  See
``research_artifacts/jul28_large_gain_anatomy_and_refinement.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field

from agent_evolve.application.calibrated_positive_gain_opportunity import (
    ObjectivePoint,
    chebyshev_excess,
    non_dominated,
)
from agent_evolve.application.sequential_market_replay import (
    MarketRecord,
    ReplaySelection,
    ReplayStepReceipt,
    _corpus_action_sha256,
    _normalized_point,
)
from agent_evolve.domain.patch import require_sha256

FRONT_PROXIMITY_ADMISSION_ID = "front_proximity_admission"
FRONT_PROXIMITY_ADMISSION_VERSION = 1
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")
_DEFINITION_DOMAIN = b"agent-evolve:front-proximity-admission:v1\x00"


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


# ``non_dominated`` and ``chebyshev_excess`` now live in the calibrated
# scorer, which consumes the same geometry directly; they are re-exported
# here unchanged so every existing import keeps resolving.


@dataclass(frozen=True, slots=True)
class FrontProximityAdmissionConfig:
    """How much of the market's anchor distribution stays selectable."""

    #: Fraction of the market's known anchors retained, best (smallest
    #: Chebyshev excess) first.  1.0 disables the restriction entirely.
    keep_fraction: float = 0.50
    #: Never shrink the support below this many candidates, so a small
    #: market can never be reduced to a single forced choice.
    minimum_support: int = 2

    def __post_init__(self) -> None:
        if (
            type(self.keep_fraction) is not float
            or not math.isfinite(self.keep_fraction)
            or not 0.0 < self.keep_fraction <= 1.0
        ):
            raise ValueError("keep_fraction must lie in (0, 1]")
        if type(self.minimum_support) is not int or self.minimum_support < 1:
            raise ValueError("minimum_support must be a positive integer")

    def to_record(self) -> dict[str, object]:
        return {
            "keep_fraction_hex": float(self.keep_fraction).hex(),
            "minimum_support": self.minimum_support,
        }


@dataclass(frozen=True, slots=True)
class FrontProximityAdmission:
    """Restrict a replay policy to the front-proximal part of its market.

    ``anchors`` maps ``action_sha256`` to that candidate's parent objective
    point in the market's own normalized metric frame.  Candidates absent
    from the map are scored at the median known excess.
    """

    inner: object
    anchors: dict[str, ObjectivePoint]
    config: FrontProximityAdmissionConfig = field(
        default_factory=FrontProximityAdmissionConfig
    )
    policy_id: str = "replay_arm.front_proximity_admission"
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not hasattr(self.inner, "select"):
            raise TypeError("inner must expose an exact select()")
        if type(self.anchors) is not dict:
            raise TypeError("anchors must be an exact mapping")
        for action_sha256, point in self.anchors.items():
            require_sha256(action_sha256, "anchors key")
            if type(point) is not tuple or not point:
                raise TypeError("anchor points must be exact tuples")
        self.config.__post_init__()
        if _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed token grammar")
        object.__setattr__(
            self,
            "definition_sha256",
            _hash(
                {
                    "arm": FRONT_PROXIMITY_ADMISSION_ID,
                    "version": FRONT_PROXIMITY_ADMISSION_VERSION,
                    "config": self.config.to_record(),
                    "inner_policy_id": str(
                        getattr(self.inner, "policy_id", "")
                    ),
                    "inner_definition_sha256": str(
                        getattr(self.inner, "definition_sha256", "")
                    ),
                    "rule": (
                        "keep the market's best keep_fraction of candidates "
                        "by parent-anchor chebyshev excess over the archive "
                        "front, unknown anchors scored at the median, never "
                        "below minimum_support"
                    ),
                }
            ),
        )

    def admissible(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
    ) -> tuple[str, ...]:
        """The retained support, outcome-blind except for revealed points."""

        support = tuple(sorted(set(selectable_action_sha256s)))
        if not support:
            raise ValueError("selectable support must be non-empty")
        if self.config.keep_fraction >= 1.0:
            return support
        metric_ids = record.metric_ids
        archive: list[ObjectivePoint] = list(record.archive_points)
        for receipt in revealed:
            candidate = record.candidate(receipt.action_sha256)
            if candidate.objectives is not None:
                archive.append(candidate.objectives)
        frozen_archive = tuple(archive)
        scored: list[tuple[str, float | None]] = []
        for action_sha256 in support:
            anchor = self.anchors.get(action_sha256)
            scored.append(
                (
                    action_sha256,
                    None
                    if anchor is None
                    else chebyshev_excess(
                        anchor, frozen_archive, metric_ids
                    ),
                )
            )
        known = sorted(
            value for _sha, value in scored if value is not None
        )
        if not known:
            return support
        median = known[len(known) // 2]
        index = int(round(self.config.keep_fraction * len(known))) - 1
        index = max(0, min(len(known) - 1, index))
        cutoff = known[index]
        kept = tuple(
            action_sha256
            for action_sha256, value in scored
            if (median if value is None else value) <= cutoff
        )
        if len(kept) < min(self.config.minimum_support, len(support)):
            # Restore in ascending excess order until the floor is met, so
            # the restriction degrades continuously instead of collapsing.
            order = sorted(
                scored,
                key=lambda item: (
                    median if item[1] is None else item[1],
                    item[0],
                ),
            )
            kept = tuple(
                action_sha256
                for action_sha256, _value in order[
                    : min(self.config.minimum_support, len(support))
                ]
            )
        return tuple(sorted(kept)) or support

    def select(
        self,
        *,
        record: MarketRecord,
        revealed: tuple[ReplayStepReceipt, ...],
        selectable_action_sha256s: tuple[str, ...],
        step_index: int,
        budget: int,
    ) -> ReplaySelection:
        kept = self.admissible(
            record=record,
            revealed=revealed,
            selectable_action_sha256s=selectable_action_sha256s,
        )
        selection = self.inner.select(
            record=record,
            revealed=revealed,
            selectable_action_sha256s=kept,
            step_index=step_index,
            budget=budget,
        )
        if type(selection) is not ReplaySelection:
            raise TypeError("inner policy must return an exact selection")
        if selection.action_sha256 not in kept:
            raise ValueError("inner policy escaped the admissible support")
        return ReplaySelection(
            action_sha256=selection.action_sha256,
            selection_propensity=selection.selection_propensity,
            evidence={
                **dict(selection.evidence),
                "admissible_support_size": len(kept),
                "offered_support_size": len(
                    set(selectable_action_sha256s)
                ),
            },
        )


def anchors_from_corpus(
    payload: dict[str, object],
) -> dict[str, ObjectivePoint]:
    """Parent anchors for one jul27 replay-corpus payload, if recorded.

    Candidates with no recorded parent objective vector are simply absent,
    which the admission rule treats as the median anchor.
    """

    market_id = str(payload["market_id"])
    axes = tuple(payload["hv_reference_point"]["axes"])  # type: ignore[index]
    metric_ids = tuple(sorted(str(axis["metric_id"]) for axis in axes))
    anchors: dict[str, ObjectivePoint] = {}
    for raw in payload["candidates"]:  # type: ignore[union-attr]
        parent = raw.get("parent")
        if not isinstance(parent, dict):
            continue
        objectives = parent.get("objectives")
        if not isinstance(objectives, dict):
            continue
        if not set(metric_ids) <= set(objectives):
            continue
        anchors[_corpus_action_sha256(market_id, raw)] = _normalized_point(
            {
                metric_id: float(objectives[metric_id])
                for metric_id in metric_ids
            },
            axes,
        )
    return anchors


__all__ = [
    "FRONT_PROXIMITY_ADMISSION_ID",
    "FRONT_PROXIMITY_ADMISSION_VERSION",
    "FrontProximityAdmission",
    "FrontProximityAdmissionConfig",
    "anchors_from_corpus",
    "chebyshev_excess",
    "non_dominated",
]

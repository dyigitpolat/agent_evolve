"""Composition portfolio selection: span the proposal sources, choose within at random.

This is a *product* policy, not an experiment arm.  It replaces the calibrated
score, the tie-break cascade and the forecast channel with one rule:

    spend the budget so that it spans the proposal sources, and pick uniformly
    inside each source.

The evidence for that rule, in one line: enforcing the family composition the
allocator was already under captures 0.801 of a clairvoyant oracle while the
model's own top-k captures 0.748, measured over 36 waves with every 4-subset of
8 enumerated exactly; and the two proposal sources are complementary in kind --
the model supplies hit rate, the numerical engine supplies the rare large hit
that carries a heavy-tailed total.  What makes a slate good is that it spans the
sources, not that it is ordered well within them.

Two properties this policy deliberately has
-------------------------------------------
**It makes no provider call.**  Selection is arithmetic over the sealed finite
contract.  A campaign running this policy spends model budget on *generation*
only.

**Its composition is derived, never hard-coded.**  Baking in the family mix
observed on five research domains would ship a constant fitted to those domains
and call it general.  Instead the caller injects
:class:`CompositionObservationSource`, which supplies realized family shares
from prior sealed campaigns; the policy apportions the budget to those shares by
largest remainder.  With no observations it falls back to *spanning*, which has
no parameters at all: one seat per distinct family before any family repeats.
Replay evidence that this transfers rather than overfits: proportions estimated
leave-one-market-out scored 0.22815 against 0.23202 for proportions taken from
the market's own realized multiset.

Determinism
-----------
"Uniform within" is drawn from a generator seeded by the request's own identity
(the finite contract identity and the call id), so a decision is reproducible
from receipts and two runs of the same campaign make the same choice.  The
policy is a pure function of its inputs; it holds no mutable state.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from agent_evolve.domain.finite_variation import FiniteVariationOption
from agent_evolve.ports.agentic_generator import (
    MetricEffectDirection,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_selection import (
    PortfolioMemberDraft,
    PortfolioSelectionRequest,
    PortfolioSelectionResult,
    pairwise_disjoint_parent_patch_witness,
    resolve_ranked_portfolio_decision,
)

COMPOSITION_PORTFOLIO_POLICY_ID = "composition_spanning_portfolio"
COMPOSITION_PORTFOLIO_POLICY_VERSION = 1

# The identity a receipt pins. It covers the rule, not the observations: the
# same policy fed different receipt-derived shares is the same policy.
COMPOSITION_PORTFOLIO_POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:composition-spanning-portfolio:v1\x00"
    b"rule=span-sources-then-uniform-within;"
    b"apportionment=largest-remainder;"
    b"shares=injected-observations-or-uniform-spanning;"
    b"provider-calls=0"
).hexdigest()


@runtime_checkable
class CompositionObservationSource(Protocol):
    """Realized family shares observed in prior sealed campaigns.

    The contract is deliberately minimal: a mapping from family token to a
    non-negative weight. Weights need not sum to one; only their ratios matter.
    An empty mapping means "no observations", and the policy falls back to
    parameter-free spanning rather than inventing a mix.
    """

    def family_shares(self) -> dict[str, float]: ...


@dataclass(frozen=True, slots=True)
class FrozenCompositionObservations:
    """A concrete observation source, e.g. built leave-one-campaign-out."""

    shares: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if type(self.shares) is not tuple:
            raise TypeError("shares must be an exact tuple")
        seen = set()
        for entry in self.shares:
            if type(entry) is not tuple or len(entry) != 2:
                raise TypeError("each share must be an exact (family, weight) pair")
            family, weight = entry
            if type(family) is not str or not family:
                raise ValueError("family must be non-empty text")
            if type(weight) is not float or weight < 0.0 or weight != weight:
                raise ValueError("weight must be a non-negative finite float")
            if family in seen:
                raise ValueError("shares repeat a family")
            seen.add(family)

    def family_shares(self) -> dict[str, float]:
        return {family: weight for family, weight in self.shares}


def _seat_apportionment(
    families: list[str],
    shares: dict[str, float],
    budget: int,
    capacity: dict[str, int],
) -> dict[str, int]:
    """Largest-remainder apportionment of `budget` seats over `families`.

    Deterministic given the inputs. Never assigns a family more seats than it
    has options, and never leaves the budget unfilled while capacity remains.
    """

    weights = {f: max(float(shares.get(f, 0.0)), 0.0) for f in families}
    total = sum(weights.values())
    if total <= 0.0:
        # no usable observations: spanning, i.e. every family weighted alike
        weights = {f: 1.0 for f in families}
        total = float(len(families))

    quota = {f: budget * weights[f] / total for f in families}
    seats = {f: min(int(quota[f]), capacity[f]) for f in families}

    # distribute the remainder by largest fractional part, then by family token
    # so the result cannot depend on dict ordering
    while sum(seats.values()) < budget:
        eligible = [f for f in families if seats[f] < capacity[f]]
        if not eligible:
            break
        eligible.sort(key=lambda f: (-(quota[f] - seats[f]), f))
        seats[eligible[0]] += 1
    return seats


def _spanning_order(
    families: list[str],
    capacity: dict[str, int],
    budget: int,
    rng: random.Random,
) -> dict[str, int]:
    """One seat per distinct family before any family repeats. No parameters."""

    order = sorted(families)
    rng.shuffle(order)
    seats = {f: 0 for f in families}
    while sum(seats.values()) < budget:
        progressed = False
        for family in order:
            if sum(seats.values()) >= budget:
                break
            if seats[family] < capacity[family]:
                seats[family] += 1
                progressed = True
        if not progressed:
            break
    return seats


def _declined_predictions(
    metric_ids: tuple[str, ...],
) -> tuple[MetricEffectPrediction, ...]:
    """An explicit non-forecast for every metric the port requires.

    The port's schema mandates a prediction per required metric. This policy has
    no forecast channel, so it declines rather than inventing a direction:
    every prediction is UNKNOWN. That is deliberate and is what makes the
    absence of a forecast channel visible in the receipt instead of hidden.
    """

    return tuple(
        MetricEffectPrediction(
            metric_id=metric_id,
            direction=MetricEffectDirection.UNKNOWN,
        )
        for metric_id in metric_ids
    )


@dataclass(frozen=True, slots=True)
class CompositionPortfolioSelectionPolicy:
    """Span the proposal sources; choose uniformly within them.

    ``observations`` is optional. Supplied, the budget is apportioned to the
    observed family shares; omitted, the policy spans families evenly, which is
    parameter-free.
    """

    observations: CompositionObservationSource | None = None

    def _rng(self, request: PortfolioSelectionRequest) -> random.Random:
        seed = hashlib.sha256(
            b"agent-evolve:composition-portfolio-draw:v1\x00"
            + request.finite_variation_contract.identity_sha256.encode("ascii")
            + b"\x00"
            + request.call_id.value.encode("ascii")
        ).digest()
        return random.Random(int.from_bytes(seed[:16], "big"))

    def _choose(
        self, request: PortfolioSelectionRequest
    ) -> tuple[FiniteVariationOption, ...]:
        contract = request.finite_variation_contract
        budget = request.portfolio_size
        rng = self._rng(request)

        by_family: dict[str, list[FiniteVariationOption]] = {}
        for option in contract.options:
            by_family.setdefault(option.family, []).append(option)
        for options in by_family.values():
            options.sort(key=lambda o: o.option_id)
        families = sorted(by_family)
        capacity = {f: len(by_family[f]) for f in families}

        chosen: list[FiniteVariationOption] = []
        taken: set[str] = set()

        # options the engine requires in the pool are honoured first: the
        # policy adds composition, it does not override a hard constraint
        required = set(request.candidate_pool_required_option_ids)
        for option in contract.options:
            if option.option_id in required and len(chosen) < budget:
                chosen.append(option)
                taken.add(option.option_id)

        remaining = budget - len(chosen)
        if remaining > 0:
            free_capacity = {
                f: sum(1 for o in by_family[f] if o.option_id not in taken)
                for f in families
            }
            shares = self.observations.family_shares() if self.observations else {}
            usable = {f: w for f, w in shares.items() if f in free_capacity}
            if usable and sum(usable.values()) > 0.0:
                seats = _seat_apportionment(
                    families, usable, remaining, free_capacity
                )
            else:
                seats = _spanning_order(families, free_capacity, remaining, rng)

            for family in sorted(families):
                pool = [o for o in by_family[family] if o.option_id not in taken]
                take = min(seats.get(family, 0), len(pool))
                if take <= 0:
                    continue
                for option in rng.sample(pool, take):
                    chosen.append(option)
                    taken.add(option.option_id)

        # top up if apportionment could not fill the budget (small catalogs)
        if len(chosen) < budget:
            pool = [o for o in contract.options if o.option_id not in taken]
            rng.shuffle(pool)
            for option in pool[: budget - len(chosen)]:
                chosen.append(option)
                taken.add(option.option_id)

        if len(chosen) != budget:
            raise ValueError(
                "composition policy could not fill the portfolio from the contract"
            )

        if request.require_pairwise_disjoint_parent_patches:
            chosen = self._project_to_legal(request, chosen)
        return tuple(chosen)

    @staticmethod
    def _project_to_legal(
        request: PortfolioSelectionRequest,
        chosen: list[FiniteVariationOption],
    ) -> list[FiniteVariationOption]:
        """Restrict the slate to one the port will accept.

        LEGALITY ONLY. The witness is the port's own feasibility search over the
        sealed parent-relative patches; the port documents it as a structural
        decoding aid that "proves how to satisfy the hard combinatorial contract
        without ranking an option by objective quality". The composition order
        is handed to it as ``preferred_option_ids``, so the projection keeps as
        much of the composition as legality allows and changes nothing else.

        This is not a quality heuristic and must never become one: the deployed
        allocator is already subject to this same rule, and adding anything that
        also picks better would change the treatment.
        """

        contract = request.finite_variation_contract
        witness = pairwise_disjoint_parent_patch_witness(
            contract,
            tuple(option.option_id for option in contract.options),
            portfolio_size=request.portfolio_size,
            min_distinct_families=request.min_distinct_families,
            preferred_option_ids=tuple(option.option_id for option in chosen),
            required_option_ids=tuple(request.candidate_pool_required_option_ids),
        )
        if witness is None:
            raise ValueError(
                "the finite contract admits no pairwise-disjoint portfolio of "
                "the requested size"
            )
        by_id = {option.option_id: option for option in contract.options}
        return [by_id[option_id] for option_id in witness]

    async def select(
        self, request: PortfolioSelectionRequest
    ) -> PortfolioSelectionResult:
        if type(request) is not PortfolioSelectionRequest:
            raise TypeError("request must be an exact PortfolioSelectionRequest")
        request.__post_init__()

        options = self._choose(request)
        card_keys = tuple(sorted({card.card_key for card in request.cards}))
        supporting = card_keys[:1] if request.require_supporting_cards else ()

        drafts = tuple(
            PortfolioMemberDraft(
                option_id=option.option_id,
                supporting_card_keys=supporting,
                effect_predictions=_declined_predictions(
                    request.required_metric_ids
                ),
                design_rationale=(
                    "composition policy: seat apportioned to proposal source "
                    f"{option.family!r}, member drawn uniformly within it; no "
                    "score, no tie-break, no forecast"
                ),
            )
            for option in options
        )
        decision = resolve_ranked_portfolio_decision(
            request,
            drafts,
            policy_id=COMPOSITION_PORTFOLIO_POLICY_ID,
            policy_version=COMPOSITION_PORTFOLIO_POLICY_VERSION,
            policy_definition_sha256=COMPOSITION_PORTFOLIO_POLICY_DEFINITION_SHA256,
        )
        # provider_free is asserted, not left to be inferred from a missing
        # telemetry field. The runtime verifies the assertion against the
        # outbound journals and fails the wave if any call was observed.
        return PortfolioSelectionResult(
            decision=decision, telemetry=None, provider_free=True
        )


__all__ = [
    "COMPOSITION_PORTFOLIO_POLICY_DEFINITION_SHA256",
    "COMPOSITION_PORTFOLIO_POLICY_ID",
    "COMPOSITION_PORTFOLIO_POLICY_VERSION",
    "CompositionObservationSource",
    "CompositionPortfolioSelectionPolicy",
    "FrozenCompositionObservations",
]

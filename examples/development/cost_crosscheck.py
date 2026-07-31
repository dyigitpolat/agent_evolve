"""Recompute what each provider call should have cost, and compare.

WHAT THIS ESTABLISHES, AND WHAT IT DOES NOT
===========================================

This is **not** independent verification, and it must not be cited as one.
Both numbers it compares come from our own journals: the token counts and the
provider's reported ``cost_usd`` are recorded by the same event, written by the
same run. A call that was never journaled at all is invisible to both sides, so
this cannot detect un-journaled spend. Closing that requires a provider-side
export -- an invoice or a usage API read -- and until one exists, the gap stays
open and should be described as open.

What it does detect, any one of which would silently corrupt every cost figure
we publish:

* **a stale or wrong price snapshot** -- prices changed, a discount appeared or
  lapsed, or the snapshot names a different endpoint than the one that served
  the call;
* **a misreported cost** -- the provider's ``cost_usd`` disagreeing with its own
  published rates for the tokens it says it processed;
* **token counts that do not match the charge** -- usage fields that drifted
  from what was billed.

It also reports something a passing check would otherwise hide: **how much of
the spend it was able to price at all**. A route with no snapshot is reported as
uncovered, never as agreeing. "No disagreement found" across a fifth of the
spend is not the same sentence as "the costs are verified", and this project
has been bitten enough times by checks that could not fail to keep those apart
in the output.

THE BILLING MODEL
=================

Fitted against 229 sealed calls on two routes that have price snapshots, and
exact to the last recorded digit on both::

    cost = (input_tokens - cache_read_tokens) * prompt_price
         +  cache_read_tokens                 * input_cache_read_price
         +  output_tokens                     * completion_price

Two facts worth stating because guessing either way is easy and wrong:

* ``reasoning_tokens`` are **already inside** ``output_tokens``. Adding them
  again overcharges by 18-45% on reasoning-heavy calls.
* ``cache_read_tokens`` are **a subset of** ``input_tokens``, billed at the
  cheaper cached rate, so they are subtracted from the prompt term rather than
  added alongside it.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Iterable, Iterator, Mapping, Optional, Sequence

__all__ = [
    "RoutePrices",
    "CallCost",
    "RouteReport",
    "CrosscheckReport",
    "load_price_snapshots",
    "expected_cost",
    "crosscheck",
    "UnpriceableCall",
    "ROUNDING_TOLERANCE",
    "MATERIAL_TOLERANCE",
]

class UnpriceableCall(ValueError):
    """The published rates do not cover something this call reports."""


#: Below this relative difference a disagreement is decimal rounding in the
#: provider's own reported figure, not a finding.
ROUNDING_TOLERANCE = Decimal("0.000001")

#: At or above this aggregate relative difference the run fails. A tenth of a
#: percent on a cost figure is already more than the precision anything is
#: reported to, so anything larger is a real disagreement about money.
MATERIAL_TOLERANCE = Decimal("0.001")

#: A ratio shared by at least this many calls is systematic -- a discount or a
#: stale price -- rather than scattered corruption. Both fail; they are named
#: differently because they need different fixes.
_SYSTEMATIC_MIN_CALLS = 3


@dataclass(frozen=True)
class RoutePrices:
    """Per-token USD prices for one resolved (model, provider) endpoint."""

    model: str
    provider: str
    prompt: Decimal
    completion: Decimal
    #: ``None`` when the endpoint publishes no cached-input rate. Harmless
    #: until a call actually reports cache reads, which is then unpriceable.
    input_cache_read: Optional[Decimal]
    source: str
    retrieved_at_utc: Optional[str] = None
    discount: Optional[float] = None

    @property
    def key(self) -> tuple:
        return (self.model, self.provider)


@dataclass(frozen=True)
class CallCost:
    """One journaled call, its reported cost, and what the prices imply."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    reported: Decimal
    expected: Decimal
    journal: str
    #: False when the journal carries no cache-token fields. Without them the
    #: cache split is unknown and ``expected`` is an upper bound, so the call
    #: is reported separately instead of counted as a disagreement.
    cache_recorded: bool = True

    @property
    def difference(self) -> Decimal:
        return self.expected - self.reported

    @property
    def relative(self) -> Decimal:
        if self.reported == 0:
            return Decimal(0) if self.expected == 0 else Decimal(1)
        return abs(self.difference) / self.reported

    @property
    def ratio(self) -> Decimal:
        """Reported over expected. 1 means the published rates explain it."""
        if self.expected == 0:
            return Decimal(1) if self.reported == 0 else Decimal(0)
        return self.reported / self.expected


@dataclass
class RouteReport:
    """Agreement for one route, split by the three things that cause it to fail.

    Keeping them apart is the point. A stale price, an incomplete journal and a
    genuinely wrong charge all show up as "the numbers differ", and they need
    three different fixes -- so reporting one number for all three would send
    whoever reads it looking in the wrong place.
    """

    model: str
    provider: str
    calls: int = 0
    reported: Decimal = Decimal(0)
    expected: Decimal = Decimal(0)
    worst: Decimal = Decimal(0)
    ratios: dict = field(default_factory=lambda: defaultdict(int))
    #: Calls whose journal carries no cache-token fields. Their cache split is
    #: unknown, so the recomputed figure is an upper bound, not a comparison.
    cache_unknown: int = 0
    cache_unknown_reported: Decimal = Decimal(0)
    cache_unknown_expected: Decimal = Decimal(0)

    @property
    def checkable_calls(self) -> int:
        return self.calls - self.cache_unknown

    @property
    def checkable_reported(self) -> Decimal:
        return self.reported - self.cache_unknown_reported

    @property
    def checkable_expected(self) -> Decimal:
        return self.expected - self.cache_unknown_expected

    @property
    def relative(self) -> Decimal:
        """Relative difference over calls that can actually be compared."""
        base = self.checkable_reported
        if base == 0:
            return Decimal(0) if self.checkable_expected == 0 else Decimal(1)
        return abs(self.checkable_expected - base) / base

    @property
    def agrees(self) -> bool:
        return self.relative < MATERIAL_TOLERANCE

    def diagnosis(self) -> str:
        """Name the shape of the disagreement, because the fixes differ."""
        if self.agrees:
            return "agrees"
        off = {r: n for r, n in self.ratios.items() if abs(r - Decimal(1)) > ROUNDING_TOLERANCE}
        if not off:
            return "disagrees in aggregate while every call agrees (check the arithmetic here)"
        dominant, count = max(off.items(), key=lambda kv: kv[1])
        if count >= _SYSTEMATIC_MIN_CALLS and len(off) <= 2:
            return (
                f"stale price: {count} call(s) charged at a consistent {dominant} of the "
                "published rate. One ratio repeated exactly is a discount or a price "
                "change the snapshot does not carry -- recapture the snapshot"
            )
        return (
            f"charges disagree: {len(off)} distinct ratios across {sum(off.values())} call(s). "
            "Scattered ratios are not explained by a price change, and this is the "
            "serious case -- the charge does not follow the rates for the tokens reported"
        )


@dataclass
class CrosscheckReport:
    """The whole comparison, including what could not be priced."""

    routes: dict = field(default_factory=dict)
    uncovered: dict = field(default_factory=lambda: defaultdict(lambda: [0, Decimal(0)]))
    unpriceable: int = 0

    @property
    def covered_calls(self) -> int:
        return sum(r.calls for r in self.routes.values())

    @property
    def covered_reported(self) -> Decimal:
        return sum((r.reported for r in self.routes.values()), Decimal(0))

    @property
    def uncovered_calls(self) -> int:
        return sum(v[0] for v in self.uncovered.values())

    @property
    def uncovered_reported(self) -> Decimal:
        return sum((v[1] for v in self.uncovered.values()), Decimal(0))

    @property
    def total_reported(self) -> Decimal:
        return self.covered_reported + self.uncovered_reported

    @property
    def coverage(self) -> Decimal:
        total = self.total_reported
        return Decimal(1) if total == 0 else self.covered_reported / total

    @property
    def cache_unknown_calls(self) -> int:
        return sum(r.cache_unknown for r in self.routes.values())

    @property
    def cache_unknown_reported(self) -> Decimal:
        return sum((r.cache_unknown_reported for r in self.routes.values()), Decimal(0))

    @property
    def agrees(self) -> bool:
        """True when every priced route agrees. Says nothing about uncovered spend."""
        return all(r.agrees for r in self.routes.values())


def load_price_snapshots(root, *, newest_only: bool = False) -> dict:
    """Load every ``*pricing_snapshot*.json`` under *root*, keyed by route.

    Returns **every** dated snapshot per route, newest first, because prices
    change and a campaign sealed in July cannot be priced with a rate captured
    in December. A charge is explained if it matches any rate we recorded for
    that route; a charge matching none is the finding.

    Using only the newest is available for callers that want to ask "are the
    current rates what we are being charged today", but it is the wrong
    question to ask of a sealed campaign, and answering it there produced two
    false "stale price" findings that were only the passage of time.
    """
    found: dict = {}
    for path in sorted(pathlib.Path(root).rglob("*pricing_snapshot*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            endpoint = doc["selected_endpoint"]
            rates = endpoint["pricing_usd_per_token"]
            prices = RoutePrices(
                model=doc["model"]["requested_slug"],
                provider=endpoint["provider_name"],
                prompt=Decimal(str(rates["prompt"])),
                completion=Decimal(str(rates["completion"])),
                input_cache_read=(
                    Decimal(str(rates["input_cache_read"]))
                    if rates.get("input_cache_read") is not None
                    else None
                ),
                source=str(path),
                retrieved_at_utc=doc.get("retrieved_at_utc"),
                discount=endpoint.get("discount"),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        found.setdefault(prices.key, []).append(prices)
    for key, values in found.items():
        values.sort(key=lambda v: v.retrieved_at_utc or "", reverse=True)
        if newest_only:
            found[key] = values[:1]
    return found


def expected_cost(response: Mapping, prices: RoutePrices) -> Decimal:
    """Cost implied by the published rates for the tokens this call reports."""
    prompt_tokens = int(response.get("input_tokens") or 0)
    completion_tokens = int(response.get("output_tokens") or 0)
    cached = int(response.get("cache_read_tokens") or 0)
    uncached = max(prompt_tokens - cached, 0)
    if cached and prices.input_cache_read is None:
        raise UnpriceableCall(
            f"{prices.model} @ {prices.provider} publishes no input_cache_read "
            f"rate, but this call reports {cached} cached input tokens"
        )
    cache_rate = prices.input_cache_read or Decimal(0)
    return uncached * prices.prompt + cached * cache_rate + completion_tokens * prices.completion


def _dedup_key(response: Mapping) -> tuple:
    """Identity of one provider call, for collapsing repeats across journals."""
    response_id = response.get("provider_response_id")
    if response_id:
        return ("id", str(response_id))
    # Without a provider id, fall back to the billed facts. Two genuinely
    # distinct calls agreeing on all of these would be indistinguishable, and
    # collapsing them understates rather than inflates -- the safer direction
    # for a check whose job is to notice money that does not add up.
    return (
        "shape",
        str(response.get("resolved_model")),
        str(response.get("resolved_provider")),
        str(response.get("cost_usd")),
        int(response.get("input_tokens") or 0),
        int(response.get("output_tokens") or 0),
        int(response.get("cache_read_tokens") or 0),
        int(response.get("latency_ns") or 0),
    )


def iter_priced_responses(paths: Iterable, *, dedup: bool = True) -> Iterator[tuple]:
    """Yield ``(response, journal)`` for every journalled call carrying a cost.

    One call is commonly recorded in more than one journal -- a queue outcome
    and an engine trace, say. Measured over the sealed corpus, 110 of 977 calls
    appear twice, so summing without collapsing them overstates spend by about
    ten percent. Deduplication is on by default for that reason; the first
    sighting wins.
    """
    seen = set()
    for path in paths:
        try:
            lines = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(row, dict):
                continue
            for candidate in (row.get("response"), row):
                if not isinstance(candidate, dict):
                    continue
                if candidate.get("cost_usd") is None:
                    continue
                if candidate.get("input_tokens") is None:
                    continue
                if dedup:
                    key = _dedup_key(candidate)
                    if key in seen:
                        break
                    seen.add(key)
                yield candidate, str(path)
                break


def crosscheck(paths: Iterable, prices_by_route: Mapping) -> CrosscheckReport:
    """Compare every journalled call against the published rates for its route."""
    report = CrosscheckReport()
    for response, journal in iter_priced_responses(paths):
        model = response.get("resolved_model")
        provider = response.get("resolved_provider")
        try:
            reported = Decimal(str(response["cost_usd"]))
        except (ValueError, ArithmeticError):
            report.unpriceable += 1
            continue
        candidates = prices_by_route.get((model, provider))
        if isinstance(candidates, RoutePrices):
            candidates = [candidates]
        if not candidates:
            slot = report.uncovered[(model, provider)]
            slot[0] += 1
            slot[1] += reported
            continue
        # The rate in force when the call was made is the one that should
        # explain it, and a sealed campaign predates today's prices. Take the
        # recorded rate that comes closest; a call explained by none of them is
        # the finding, and one explained by an older one is simply older.
        implied = None
        best = None
        for prices in candidates:
            try:
                value = expected_cost(response, prices)
            except UnpriceableCall:
                continue
            gap = abs(value - reported)
            if best is None or gap < best:
                best, implied, matched = gap, value, prices
        if implied is None:
            slot = report.uncovered[(model, provider)]
            slot[0] += 1
            slot[1] += reported
            continue
        prices = matched
        call = CallCost(
            model=model,
            provider=provider,
            input_tokens=int(response.get("input_tokens") or 0),
            output_tokens=int(response.get("output_tokens") or 0),
            cache_read_tokens=int(response.get("cache_read_tokens") or 0),
            reported=reported,
            expected=implied,
            journal=journal,
            cache_recorded=response.get("cache_read_tokens") is not None,
        )
        route = report.routes.setdefault((model, provider), RouteReport(model, provider))
        route.calls += 1
        route.reported += call.reported
        route.expected += call.expected
        if not call.cache_recorded:
            route.cache_unknown += 1
            route.cache_unknown_reported += call.reported
            route.cache_unknown_expected += call.expected
            continue
        route.worst = max(route.worst, call.relative)
        route.ratios[call.ratio.quantize(Decimal("0.0001"))] += 1
    return report


def format_report(report: CrosscheckReport) -> str:
    """Human-readable summary, coverage first because it bounds everything else."""
    out = ["cost cross-check: recomputed charges vs provider-reported cost", ""]
    out.append(
        f"  priced   {report.covered_calls:>5} calls  ${report.covered_reported:>10.4f}"
    )
    out.append(
        f"  unpriced {report.uncovered_calls:>5} calls  ${report.uncovered_reported:>10.4f}"
        "   (no price snapshot for the route)"
    )
    if report.cache_unknown_calls:
        out.append(
            f"  of which {report.cache_unknown_calls} call(s) "
            f"(${report.cache_unknown_reported:.4f}) journal no cache-token fields, so "
            "their\n           cache split is unknown and they are excluded from the "
            "comparison rather\n           than counted as disagreeing"
        )
    out.append(f"  coverage {report.coverage * 100:.1f}% of reported spend")
    out.append("")
    for (model, provider), route in sorted(report.routes.items(), key=lambda kv: -kv[1].reported):
        mark = "ok  " if route.agrees else "FAIL"
        out.append(
            f"  [{mark}] {model} @ {provider}: {route.checkable_calls} comparable "
            f"of {route.calls} calls  reported ${route.checkable_reported:.6f}  "
            f"expected ${route.checkable_expected:.6f}  rel {route.relative:.2e}"
        )
        if not route.agrees:
            out.append(f"           {route.diagnosis()}")
    if report.uncovered:
        out.append("")
        out.append("  routes with no price snapshot, so nothing was checked:")
        for (model, provider), (calls, cost) in sorted(
            report.uncovered.items(), key=lambda kv: -kv[1][1]
        ):
            out.append(f"    {model} @ {provider}: {calls} calls  ${cost:.4f}")
    out.append("")
    out.append(
        "  This is not independent verification: both figures come from our own\n"
        "  journals, so a call that was never journaled is invisible to it.\n"
        "  Standing limitations: research_artifacts/COST_ACCOUNTING_LIMITATIONS.md"
    )
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cost_crosscheck",
        description=(
            "Recompute each journalled provider call from published per-token "
            "prices and compare against the cost the provider reported."
        ),
    )
    parser.add_argument("root", help="directory to scan for journals and price snapshots")
    parser.add_argument(
        "--journal-glob",
        default="*.jsonl",
        help="journal filename pattern (default: every .jsonl)",
    )
    parser.add_argument(
        "--allow-uncovered",
        action="store_true",
        help="do not fail merely because some routes have no price snapshot",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    prices = load_price_snapshots(root)
    if not prices:
        print(f"no pricing snapshots found under {root}")
        return 2
    report = crosscheck(sorted(root.rglob(args.journal_glob)), prices)
    print(format_report(report))

    if not report.agrees:
        print("\nFAILED: recomputed cost disagrees with the reported cost.")
        return 1
    if report.uncovered and not args.allow_uncovered:
        print(
            "\nFAILED: part of the spend has no price snapshot and was not checked. "
            "Capture snapshots for those routes, or pass --allow-uncovered to "
            "record that the gap is known."
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

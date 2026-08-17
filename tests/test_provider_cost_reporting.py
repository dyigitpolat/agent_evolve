"""What a run cost, derived and attributed rather than left null.

A credentialed run reported 27 calls and 41,564 tokens with `cost_usd: null`,
while the price for that exact route was sitting in `MODEL_PRICES_PER_MTOK` --
the same table the CLI echoes before it spends anything. The number was
knowable and simply not computed.

Cost is now derived, and `reported_by` carries the split provenance, because the
two halves come from different places: the provider counted the tokens, this
package priced them. An unpriced route still reports `None`, since a cost that
cannot be derived must read as unknown and never as zero.
"""

from __future__ import annotations

from decimal import Decimal

from agent_evolve.api import _priced_usage
from agent_evolve.core.results import ProviderUsageSummary
from agent_evolve.settings import MODEL_PRICES_PER_MTOK, _DEFAULT_MODEL


def test_the_default_route_is_priced_and_the_arithmetic_is_exact():
    cost, reporter = _priced_usage(_DEFAULT_MODEL, 19_542, 22_022)
    per_m_in, per_m_out = MODEL_PRICES_PER_MTOK[_DEFAULT_MODEL]
    expected = (
        Decimal(str(per_m_in)) * 19_542 + Decimal(str(per_m_out)) * 22_022
    ) / Decimal(1_000_000)
    assert Decimal(cost) == expected.quantize(Decimal("0.000001"))
    assert "openrouter response usage" in reporter
    assert "price table" in reporter, (
        "the reporter must say the cost was derived, not billed"
    )


def test_an_unpriced_route_reports_unknown_and_not_zero():
    cost, reporter = _priced_usage("some/route-we-never-priced", 1000, 1000)
    assert cost is None
    assert "price table" not in reporter


def test_zero_tokens_on_a_priced_route_is_a_real_zero():
    cost, _ = _priced_usage(_DEFAULT_MODEL, 0, 0)
    assert Decimal(cost) == Decimal("0")


def test_the_summary_accepts_a_derived_cost_with_its_reporter():
    cost, reporter = _priced_usage(_DEFAULT_MODEL, 100, 100)
    summary = ProviderUsageSummary(
        calls=1, input_tokens=100, output_tokens=100,
        cost_usd=cost, model=_DEFAULT_MODEL, reported_by=reporter,
    )
    assert summary.cost_is_known
    assert not summary.provider_free

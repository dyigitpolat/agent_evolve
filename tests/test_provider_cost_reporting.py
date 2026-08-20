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

import pytest

from agent_evolve.api import _priced_usage
from agent_evolve.core.results import ProviderUsageSummary
from agent_evolve.settings import MODEL_PRICES_PER_MTOK, _DEFAULT_MODEL, model_price


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


@pytest.mark.parametrize(
    "tier, price",
    [("luna", (0.10, 0.60)), ("terra", (1.00, 6.00)), ("sol", (5.00, 30.00))],
)
def test_every_tier_of_the_ladder_is_priced_in_both_spellings(tier, price):
    """A run is priced by the name the caller wrote, so both names must resolve.

    The ladder spans three routes and a caller may name any of them either as
    the package's prefixed form or as the bare id. A tier priced under one
    spelling and unpriced under the other reports `cost_usd: null` for no
    reason a reader could guess from the run.
    """
    bare = f"openai/gpt-5.6-{tier}"
    assert model_price(bare) == price
    assert model_price(f"openrouter:{bare}") == price


def test_the_terra_and_sol_rungs_derive_a_cost_rather_than_reporting_unknown():
    cost, reporter = _priced_usage("openai/gpt-5.6-terra", 1_000_000, 1_000_000)
    assert Decimal(cost) == Decimal("7.000000")
    assert "price table" in reporter
    cost, _ = _priced_usage("openrouter:openai/gpt-5.6-sol", 1_000_000, 1_000_000)
    assert Decimal(cost) == Decimal("35.000000")


def test_the_summary_accepts_a_derived_cost_with_its_reporter():
    cost, reporter = _priced_usage(_DEFAULT_MODEL, 100, 100)
    summary = ProviderUsageSummary(
        calls=1, input_tokens=100, output_tokens=100,
        cost_usd=cost, model=_DEFAULT_MODEL, reported_by=reporter,
    )
    assert summary.cost_is_known
    assert not summary.provider_free

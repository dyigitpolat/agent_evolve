"""The cost cross-check, on synthetic journals.

corpus-free-by-construction: every fixture here is built in tmp_path, so these
run offline and need no research corpus.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from examples.development.cost_crosscheck import (
    MATERIAL_TOLERANCE,
    crosscheck,
    expected_cost,
    iter_priced_responses,
    load_price_snapshots,
)

PROMPT = "0.000001"
COMPLETION = "0.000002"
CACHE_READ = "0.0000001"


def write_snapshot(root, *, model="m/one", provider="P", retrieved="2026-07-15T00:00:00Z",
                   prompt=PROMPT, completion=COMPLETION, cache_read=CACHE_READ, name=None):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    # The loader globs *pricing_snapshot*.json, so every fixture must match it.
    stem = (name or provider.lower()).removesuffix(".json")
    path = data / f"{stem}_pricing_snapshot.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "retrieved_at_utc": retrieved,
        "model": {"requested_slug": model},
        "selected_endpoint": {
            "provider_name": provider,
            "pricing_usd_per_token": {
                "prompt": prompt, "completion": completion, "input_cache_read": cache_read,
            },
        },
    }))
    return path


def write_journal(root, rows, name="queue_outcomes.jsonl"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps({"response": r}) for r in rows))
    return path


def call(cost, *, inp=1000, out=500, cache=0, model="m/one", provider="P", rid=None, with_cache=True):
    r = {
        "resolved_model": model, "resolved_provider": provider,
        "input_tokens": inp, "output_tokens": out, "cost_usd": str(cost),
    }
    if with_cache:
        r["cache_read_tokens"] = cache
        r["cache_write_tokens"] = 0
    if rid:
        r["provider_response_id"] = rid
    return r


# -- the billing model ------------------------------------------------------


def test_reasoning_tokens_are_not_charged_again(tmp_path):
    """Fitted on real data: reasoning sits inside output_tokens already."""
    write_snapshot(tmp_path)
    prices = load_price_snapshots(tmp_path)[("m/one", "P")][0]
    plain = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 0}
    with_reasoning = dict(plain, reasoning_tokens=400)
    assert expected_cost(plain, prices) == expected_cost(with_reasoning, prices)


def test_cache_reads_are_a_discount_on_input_not_an_addition(tmp_path):
    write_snapshot(tmp_path)
    prices = load_price_snapshots(tmp_path)[("m/one", "P")][0]
    uncached = expected_cost({"input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 0}, prices)
    cached = expected_cost({"input_tokens": 1000, "output_tokens": 0, "cache_read_tokens": 1000}, prices)
    assert cached < uncached
    assert cached == Decimal(1000) * Decimal(CACHE_READ)


# -- agreement and disagreement --------------------------------------------


def test_a_correctly_charged_run_agrees_and_passes(tmp_path):
    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    write_journal(tmp_path, [call(exact, rid="a"), call(exact, rid="b")])
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    assert report.agrees
    assert report.covered_calls == 2
    assert report.coverage == 1


def test_a_consistent_ratio_is_named_a_stale_price(tmp_path):
    """The real finding: 46 sealed calls charged at exactly 0.939 of list."""
    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    discounted = (exact * Decimal("0.939")).quantize(Decimal("0.00000001"))
    write_journal(tmp_path, [call(discounted, rid=f"d{i}") for i in range(6)])
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    route = report.routes[("m/one", "P")]
    assert not route.agrees
    assert "stale price" in route.diagnosis()


def test_scattered_ratios_are_named_the_serious_case(tmp_path):
    write_snapshot(tmp_path)
    base = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    rows = [call((base * Decimal(f"0.{80 + i}")).quantize(Decimal("0.00000001")), rid=f"s{i}")
            for i in range(6)]
    write_journal(tmp_path, rows)
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    route = report.routes[("m/one", "P")]
    assert not route.agrees
    assert "charges disagree" in route.diagnosis()


def test_rounding_alone_does_not_fail(tmp_path):
    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    nudged = exact + Decimal("0.0000000001")
    write_journal(tmp_path, [call(nudged, rid="r")])
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    assert report.agrees, "a rounding-sized difference must not be a finding"


# -- the things a passing check must not hide ------------------------------


def test_a_route_without_a_snapshot_is_uncovered_never_agreeing(tmp_path):
    """The check must not report success over spend it never priced."""
    write_snapshot(tmp_path)
    write_journal(tmp_path, [
        call(Decimal("0.002"), rid="x", model="m/one", provider="P"),
        call(Decimal("99.0"), rid="y", model="other/model", provider="Elsewhere"),
    ])
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    assert ("other/model", "Elsewhere") in report.uncovered
    assert report.uncovered_reported == Decimal("99.0")
    assert report.coverage < Decimal("0.01")


def test_calls_without_cache_fields_are_excluded_not_failed(tmp_path):
    """An incomplete journal is a different defect from a wrong charge."""
    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    write_journal(tmp_path, [
        call(exact, rid="ok"),
        # Heavily cached in reality, but the journal says nothing about it, so
        # the recomputed figure is an upper bound rather than a comparison.
        call(exact / 4, rid="nocache", with_cache=False),
    ])
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    route = report.routes[("m/one", "P")]
    assert route.calls == 2
    assert route.cache_unknown == 1
    assert route.checkable_calls == 1
    assert route.agrees, "the unknown-cache call must not be counted as a disagreement"


def test_one_call_in_two_journals_is_counted_once(tmp_path):
    """110 of 977 sealed calls appear twice; summing blind overstates by ~10%."""
    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    row = call(exact, rid="shared")
    write_journal(tmp_path / "run", [row], name="queue_outcomes.jsonl")
    write_journal(tmp_path / "run", [row], name="engine_traces.jsonl")
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    assert report.covered_calls == 1
    assert report.covered_reported == exact


def test_deduplication_can_be_switched_off(tmp_path):
    write_snapshot(tmp_path)
    row = call(Decimal("0.002"), rid="shared")
    write_journal(tmp_path / "run", [row], name="queue_outcomes.jsonl")
    write_journal(tmp_path / "run", [row], name="engine_traces.jsonl")
    paths = sorted(tmp_path.rglob("*.jsonl"))
    assert len(list(iter_priced_responses(paths, dedup=False))) == 2
    assert len(list(iter_priced_responses(paths, dedup=True))) == 1


# -- snapshot selection -----------------------------------------------------


def test_every_dated_snapshot_is_kept_newest_first(tmp_path):
    """A sealed campaign cannot be priced with a rate captured afterwards.

    Keeping only the newest produced two false "stale price" findings that
    were nothing but the passage of time, so every era is retained and a
    charge is explained if any recorded rate accounts for it.
    """
    write_snapshot(tmp_path, retrieved="2026-07-01T00:00:00Z", prompt="0.000009", name="old")
    write_snapshot(tmp_path, retrieved="2026-07-20T00:00:00Z", prompt="0.000001", name="new")
    prices = load_price_snapshots(tmp_path)[("m/one", "P")]
    assert [p.prompt for p in prices] == [Decimal("0.000001"), Decimal("0.000009")]
    newest = load_price_snapshots(tmp_path, newest_only=True)[("m/one", "P")]
    assert len(newest) == 1 and newest[0].prompt == Decimal("0.000001")


def test_a_call_priced_at_a_superseded_rate_still_agrees(tmp_path):
    """The StreamLake case: 84 sealed calls charged at the rate then in force."""
    write_snapshot(tmp_path, retrieved="2026-07-01T00:00:00Z", prompt="0.000009",
                   completion="0.000009", name="old")
    write_snapshot(tmp_path, retrieved="2026-07-20T00:00:00Z", prompt="0.000001",
                   completion="0.000001", name="new")
    old_price = (Decimal(1000) + Decimal(500)) * Decimal("0.000009")
    write_journal(tmp_path, [call(old_price, rid=f"o{i}") for i in range(4)])
    report = crosscheck(sorted(tmp_path.rglob("*.jsonl")), load_price_snapshots(tmp_path))
    assert report.agrees, "a charge at a superseded rate is history, not a defect"


def test_a_malformed_snapshot_is_skipped_not_fatal(tmp_path):
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "broken_pricing_snapshot.json").write_text("{not json")
    write_snapshot(tmp_path)
    assert ("m/one", "P") in load_price_snapshots(tmp_path)


# -- material disagreement fails loudly ------------------------------------


def test_material_disagreement_exits_non_zero(tmp_path):
    from examples.development.cost_crosscheck import main

    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    write_journal(tmp_path, [call(exact * Decimal("0.5"), rid=f"m{i}") for i in range(4)])
    assert main([str(tmp_path)]) == 1


def test_uncovered_spend_also_exits_non_zero_unless_acknowledged(tmp_path):
    from examples.development.cost_crosscheck import main

    write_snapshot(tmp_path)
    exact = Decimal(1000) * Decimal(PROMPT) + Decimal(500) * Decimal(COMPLETION)
    write_journal(tmp_path, [
        call(exact, rid="ok"),
        call(Decimal("5.0"), rid="elsewhere", model="other/model", provider="Elsewhere"),
    ])
    assert main([str(tmp_path)]) == 1
    assert main([str(tmp_path), "--allow-uncovered"]) == 0


def test_tolerance_is_tighter_than_any_figure_we_publish():
    assert MATERIAL_TOLERANCE <= Decimal("0.001")


# -- endpoints that publish no cached-input rate ---------------------------


def test_a_route_without_a_cache_rate_still_prices_uncached_calls(tmp_path):
    """DeepInfra publishes no input_cache_read rate; the route is still priceable."""
    from examples.development.cost_crosscheck import UnpriceableCall, expected_cost

    write_snapshot(tmp_path)
    prices = load_price_snapshots(tmp_path)[("m/one", "P")][0]
    no_cache = type(prices)(
        model=prices.model, provider=prices.provider, prompt=prices.prompt,
        completion=prices.completion, input_cache_read=None, source=prices.source,
    )
    plain = {"input_tokens": 1000, "output_tokens": 500, "cache_read_tokens": 0}
    assert expected_cost(plain, no_cache) == Decimal(1000) * prices.prompt + Decimal(500) * prices.completion
    with pytest.raises(UnpriceableCall, match="input_cache_read"):
        expected_cost({**plain, "cache_read_tokens": 10}, no_cache)


# -- the shared collapse -----------------------------------------------------


def test_the_dedup_key_prefers_the_provider_id(tmp_path):
    from examples.development.cost_crosscheck import _dedup_key

    a = call(Decimal("1"), rid="same")
    b = call(Decimal("2"), inp=99, rid="same")
    assert _dedup_key(a) == _dedup_key(b), "one provider id is one call"


def test_without_a_provider_id_the_billed_facts_identify_the_call(tmp_path):
    from examples.development.cost_crosscheck import _dedup_key

    a = call(Decimal("1"), inp=10, out=20)
    b = call(Decimal("1"), inp=10, out=20)
    c = call(Decimal("1"), inp=11, out=20)
    assert _dedup_key(a) == _dedup_key(b)
    assert _dedup_key(a) != _dedup_key(c)

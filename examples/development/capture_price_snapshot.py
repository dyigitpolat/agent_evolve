"""Capture a dated per-token price snapshot for one OpenRouter route.

Writes the same ``schema_version: 1`` shape the existing snapshots use, so
``cost_crosscheck.load_price_snapshots`` reads them without special cases.

**A route that cannot be priced is recorded, not skipped.** If the provider no
longer serves the model, the snapshot records that explicitly -- with the
providers that *are* available at capture time -- under
``route_unavailable``. "The provider stopped publishing these rates" and
"nobody ever captured them" are different problems and only one is fixable, so
they must not share an undifferentiated bucket.

This reads price metadata only. It is not an inference call and costs nothing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Optional, Sequence

__all__ = ["capture", "build_snapshot", "fetch_endpoints", "snapshot_filename"]

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
CATALOG_URL = "https://openrouter.ai/api/v1/models"

#: Price fields we carry through verbatim, as strings, so no float rounding is
#: introduced between the provider's published rate and our arithmetic.
_PRICE_FIELDS = (
    ("prompt", "prompt"),
    ("completion", "completion"),
    ("input_cache_read", "input_cache_read"),
    ("input_cache_write", "input_cache_write"),
)


def fetch_endpoints(slug: str, *, timeout: int = 60) -> dict:
    """Read the published endpoint list for one model slug."""
    url = ENDPOINTS_URL.format(slug=slug)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _price_block(pricing: dict) -> dict:
    out = {}
    for ours, theirs in _PRICE_FIELDS:
        value = pricing.get(theirs)
        if value is None:
            continue
        out[ours] = str(value)
    return out


def build_snapshot(payload: dict, *, slug: str, provider: str) -> dict:
    """Turn an endpoints payload into a snapshot for one provider."""
    data = payload.get("data") or {}
    endpoints = data.get("endpoints") or []
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = {
        "schema_version": 1,
        "retrieved_at_utc": retrieved,
        "source_url": ENDPOINTS_URL.format(slug=slug),
        "model_catalog_source_url": CATALOG_URL,
        "model": {
            "requested_slug": slug,
            "canonical_slug": data.get("id"),
            "context_length": data.get("context_length"),
        },
    }

    match = next(
        (e for e in endpoints if str(e.get("provider_name", "")).lower() == provider.lower()),
        None,
    )
    if match is None:
        # Recorded rather than skipped. Whoever reads this later needs to be
        # able to tell "the rates are gone" from "nobody looked".
        base["route_unavailable"] = {
            "requested_provider": provider,
            "reason": "the provider does not serve this model at capture time",
            "providers_available_at_capture": sorted(
                str(e.get("provider_name")) for e in endpoints if e.get("provider_name")
            ),
        }
        return base

    pricing = match.get("pricing") or {}
    prices = _price_block(pricing)
    # Only the rates every call consumes are required. A provider that offers
    # no prompt caching simply publishes no cache-read rate, and calling that
    # route "unavailable" would hide a fully priceable endpoint behind the same
    # word used for one that genuinely stopped serving the model.
    missing = [n for n in ("prompt", "completion") if n not in prices]
    if missing:
        base["route_unavailable"] = {
            "requested_provider": provider,
            "reason": f"the endpoint publishes no {', '.join(missing)} rate",
            "published_pricing_fields": sorted(pricing),
        }
        return base
    if "input_cache_read" not in prices:
        base["no_cache_read_rate"] = (
            "this endpoint publishes no input_cache_read rate; calls reporting "
            "cache reads on it cannot be priced and are reported as such"
        )

    base["selected_endpoint"] = {
        "provider_name": match.get("provider_name"),
        "provider_request_slug": match.get("tag") or match.get("provider_request_slug"),
        "name": match.get("name"),
        "endpoint_tag": match.get("tag"),
        "pricing_usd_per_token": prices,
        "discount": match.get("discount", 0),
        "quantization": match.get("quantization") or "unknown",
        "uptime_last_30m_at_retrieval": match.get("uptime_last_30m"),
    }
    return base


def snapshot_filename(slug: str, provider: str, *, retrieved: Optional[str] = None) -> str:
    stamp = (retrieved or datetime.now(timezone.utc).strftime("%Y%m%d"))[:10].replace("-", "")
    safe = slug.replace("/", "_").replace("-", "_").replace(".", "_")
    return f"openrouter_{safe}_{provider.lower().replace(' ', '_')}_pricing_snapshot_{stamp}.json"


def capture(slug: str, provider: str, out_dir) -> pathlib.Path:
    """Fetch, build and write one snapshot. Returns the path written."""
    payload = fetch_endpoints(slug)
    snapshot = build_snapshot(payload, slug=slug, provider=provider)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / snapshot_filename(slug, provider)
    path.write_text(json.dumps(snapshot, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="capture_price_snapshot",
        description="Capture a dated per-token price snapshot for an OpenRouter route.",
    )
    parser.add_argument("--out", required=True, help="directory to write snapshots into")
    parser.add_argument(
        "--route",
        action="append",
        required=True,
        metavar="SLUG@PROVIDER",
        help="e.g. openai/gpt-5.6-luna@OpenAI (repeatable)",
    )
    args = parser.parse_args(argv)

    failures = 0
    for spec in args.route:
        if "@" not in spec:
            print(f"  bad route {spec!r}; use SLUG@PROVIDER")
            failures += 1
            continue
        slug, _, provider = spec.rpartition("@")
        try:
            path = capture(slug, provider, args.out)
        except (urllib.error.URLError, OSError, ValueError) as error:
            print(f"  {spec}: could not capture ({type(error).__name__}: {error})")
            failures += 1
            continue
        doc = json.loads(path.read_text())
        if "route_unavailable" in doc:
            info = doc["route_unavailable"]
            print(f"  {spec}: UNAVAILABLE -- {info['reason']}")
            if info.get("providers_available_at_capture"):
                print(f"      providers serving it: {', '.join(info['providers_available_at_capture'])}")
        else:
            rates = doc["selected_endpoint"]["pricing_usd_per_token"]
            print(f"  {spec}: prompt={rates['prompt']} completion={rates['completion']}")
        print(f"      -> {path.name}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

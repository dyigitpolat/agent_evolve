#!/usr/bin/env python3
"""Build a comparable qualification matrix from finalized model canaries."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path
from typing import Any


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise TypeError(f"{path} must contain an object")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def analyze_canary(run_dir: Path, profile_name: str) -> dict[str, Any]:
    root = run_dir.expanduser().resolve(strict=True)
    summary = _object(root / "summary.json")
    outcomes = _rows(root / "queue_outcomes.jsonl")
    progress_path = root / "stream_progress.jsonl"
    attempts = [item for row in outcomes for item in row["attempts"]]
    responses = summary["responses"]
    costs = [Decimal(value["cost_usd"]) for value in responses]
    service_ns = sum(int(value["service_time_ns"]) for value in attempts)
    return {
        "schema_version": 1,
        "profile_name": profile_name,
        "run_id": root.name,
        "status": summary["status"],
        "profile_sha256": summary["profile"]["profile_sha256"],
        "requested_model": summary["profile"]["requested_model"],
        "provider_only": summary["profile"]["provider_options"]["only"],
        "resolved_providers": sorted(
            {value["resolved_provider"] for value in responses}
        ),
        "reasoning": summary["profile"]["reasoning"],
        "max_output_tokens": summary["profile"]["max_output_tokens"],
        "logical_calls": len(outcomes),
        "successful_logical_calls": sum(row["status"] == "succeeded" for row in outcomes),
        "physical_attempts": len(attempts),
        "retry_attempts": sum(value["status"] != "succeeded" for value in attempts),
        "failed_attempt_service_s": sum(
            int(value["service_time_ns"])
            for value in attempts
            if value["status"] != "succeeded"
        )
        / 1e9,
        "total_attempt_service_s": service_ns / 1e9,
        "input_tokens": sum(int(value["input_tokens"]) for value in responses),
        "output_tokens": sum(int(value["output_tokens"]) for value in responses),
        "reasoning_tokens": sum(int(value["reasoning_tokens"]) for value in responses),
        "cost_usd": str(sum(costs, Decimal(0))),
        "progress_record_count": sum(1 for _ in progress_path.open(encoding="utf-8")),
        "progress_file_bytes": progress_path.stat().st_size,
        "health": summary["health"],
        "campaign_promotion": (
            "hold_for_production_gate"
            if len(attempts) > len(outcomes) or progress_path.stat().st_size > 16 * 1024 * 1024
            else "eligible_for_production_gate"
        ),
    }


def _flat(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else item)
        for key, item in value.items()
        if key not in {"schema_version", "health"}
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    spec = _object(args.spec.resolve(strict=True))
    rows = [
        analyze_canary(Path(value["run_dir"]), value["profile_name"])
        for value in spec["runs"]
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"schema_version": 1, "runs": rows}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    flat = [_flat(value) for value in rows]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"run_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

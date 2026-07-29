"""Measure what a run actually sent to a provider, instead of asserting it.

A receipt that says ``"provider_calls": 0`` proves nothing when the 0 is a
literal, and neither does ``"zero_provider_calls": PROVIDER_CALLS == 0`` when
``PROVIDER_CALLS`` is a module constant set to 0 -- that expression cannot
evaluate to anything but ``True``. Assertions that cannot fail read exactly like
assertions that were checked, which is worse than having none.

The fix has two halves, and the second is the one that is easy to miss:

1. Count the journal rather than declaring the count. :func:`measure_provider_usage`
   reads the outcome and outbound journals a run has on disk and reports what is
   in them.
2. **Create the journal even when nothing will be written to it.** You cannot
   measure the absence of something you never instrumented, so a provider-free
   run that simply omits ``queue_outcomes.jsonl`` leaves no evidence at all --
   its silence is indistinguishable from an un-journaled call. Such a run must
   call :func:`ensure_declared_journals` at the start, so that an empty file at
   the end is a measured zero.

A journal a run declares in its manifest and never writes is treated as a fault,
not as a zero: :class:`DeclaredJournalMissing` fails the run loudly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

#: Declared-journal roles whose rows represent provider traffic.
OUTCOME_ROLE = "outcome"
OUTBOUND_ROLE = "outbound"


class DeclaredJournalMissing(RuntimeError):
    """A run declared a journal in its manifest and never created it."""


@dataclass(frozen=True)
class ProviderUsage:
    """Measured provider traffic. Every field is counted, none is declared."""

    provider_calls: int
    outcome_rows: int
    outbound_rows: int
    input_tokens: int
    output_tokens: int
    cost_usd: str
    journal_rows: Mapping[str, int]

    @property
    def provider_free(self) -> bool:
        """True only because the journals were read and were empty."""
        return self.provider_calls == 0 and self.outcome_rows == 0 and self.outbound_rows == 0

    def to_record(self) -> Dict[str, Any]:
        """A sealable record. ``measured_from`` names what was actually read."""
        return {
            "provider_calls": self.provider_calls,
            "outcome_rows": self.outcome_rows,
            "outbound_rows": self.outbound_rows,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "provider_free": self.provider_free,
            "measured_from": dict(sorted(self.journal_rows.items())),
        }


def ensure_declared_journals(
    run_dir: "str | Path", declared: Mapping[str, str]
) -> Tuple[str, ...]:
    """Create every declared journal that does not exist yet, empty.

    Call this when a run starts. A provider-free run that never touches a
    provider still ends with an empty ``queue_outcomes.jsonl``, and that empty
    file is the difference between "we measured zero" and "we never looked".

    Returns the names created.
    """
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    for role, filename in sorted(declared.items()):
        path = directory / filename
        if not path.exists():
            path.touch()
            created.append(role)
    return tuple(created)


def measure_provider_usage(
    run_dir: "str | Path", declared: Mapping[str, str]
) -> ProviderUsage:
    """Count provider traffic from the journals a run declared.

    Raises :class:`DeclaredJournalMissing` if a declared journal is absent --
    that is an instrumentation fault, and reporting it as zero would launder a
    missing sink into a provider-free claim.
    """
    directory = Path(run_dir)
    missing = [
        f"{role} ({filename})"
        for role, filename in sorted(declared.items())
        if not (directory / filename).is_file()
    ]
    if missing:
        raise DeclaredJournalMissing(
            f"{directory}: declared but never written: {', '.join(missing)}. "
            "A journal that was never created cannot evidence zero provider calls."
        )

    rows_by_role = {
        role: _rows(directory / filename) for role, filename in declared.items()
    }
    outcome_rows = rows_by_role.get(OUTCOME_ROLE, [])
    outbound_rows = rows_by_role.get(OUTBOUND_ROLE, [])

    calls = 0
    input_tokens = 0
    output_tokens = 0
    cost = Decimal(0)
    for row in outcome_rows:
        response = row.get("response") or {}
        if response:
            calls += 1
        input_tokens += int(response.get("input_tokens") or 0)
        output_tokens += int(response.get("output_tokens") or 0)
        if response.get("cost_usd") is not None:
            cost += Decimal(str(response["cost_usd"]))

    return ProviderUsage(
        provider_calls=calls,
        outcome_rows=len(outcome_rows),
        outbound_rows=len(outbound_rows),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=format(cost, "f"),
        journal_rows={role: len(rows) for role, rows in sorted(rows_by_role.items())},
    )


def _rows(path: Path) -> list:
    """Parse a JSONL journal. An empty file is zero rows, not a missing file."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


__all__ = [
    "DeclaredJournalMissing",
    "OUTBOUND_ROLE",
    "OUTCOME_ROLE",
    "ProviderUsage",
    "ensure_declared_journals",
    "measure_provider_usage",
]

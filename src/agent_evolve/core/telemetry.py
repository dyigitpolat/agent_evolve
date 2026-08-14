"""What each guidance mechanism actually did, surfaced with the result.

The seam objects count their own behaviour (calls, rejections, authoring
attempts) and always have — but the counters lived as attributes on closures
the caller discarded, so no run could report what its guidance did. A number
that is counted but unreachable might as well not exist: the difference
between "guidance did not help" and "guidance never arrived" is exactly these
counters, and it must be readable off the ``SearchResult``.

The contract is deliberately small. Any seam object may carry:

``telemetry``    an object whose ``as_dict()`` returns integer counters;
``mechanism``    a short name for what the seam decides (``"chooser"``);
``authored_by``  who authored the decisions — ``"llm"``, ``"rule"``, or
                 ``"none"``.

:func:`harvest_telemetry` gathers whatever is present and skips whatever is
not, so the unguided path reports an empty mechanism list rather than nothing
at all — a measured zero, distinguishable from "nobody looked".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

__all__ = ["MechanismTelemetry", "RunTelemetry", "harvest_telemetry"]


@dataclass(frozen=True)
class MechanismTelemetry:
    """One mechanism's counters, labelled with who authored its decisions."""

    mechanism: str
    authored_by: str
    counters: Mapping[str, int]


@dataclass(frozen=True)
class RunTelemetry:
    """A run's mechanism counters plus its evaluation ledger.

    ``real_evaluations`` counts evaluator invocations that were charged
    against the budget. ``virtual_evaluations`` counts surrogate predictions —
    free by construction, and reported separately precisely so the two can
    never be conflated in a budget claim.

    ``proxy_evaluations`` counts calls to a problem's CHEAPER evaluation
    fidelity (``Problem.evaluate_proxy``). They are not free — they burn real
    evaluator seconds — and they are not charged: the budget a claim is
    denominated in counts full-fidelity evaluations. A campaign that spends
    them must therefore report them, which is why they have a field of their
    own here rather than a line in someone's log.
    """

    mechanisms: Tuple[MechanismTelemetry, ...] = ()
    real_evaluations: int = 0
    virtual_evaluations: int = 0
    proxy_evaluations: int = 0


def harvest_telemetry(
    sources: Iterable[Optional[Any]],
    *,
    real_evaluations: int = 0,
    virtual_evaluations: int = 0,
    proxy_evaluations: int = 0,
) -> RunTelemetry:
    """Collect telemetry from whichever *sources* carry it.

    ``None`` entries and objects without a usable ``telemetry.as_dict()`` are
    skipped silently: absence of counters is a legitimate state (the random
    chooser, the statistical prior), not an error.
    """

    mechanisms = []
    for source in sources:
        if source is None:
            continue
        counter = getattr(source, "telemetry", None)
        as_dict = getattr(counter, "as_dict", None)
        if not callable(as_dict):
            continue
        counters = {str(k): int(v) for k, v in dict(as_dict()).items()}
        mechanisms.append(
            MechanismTelemetry(
                mechanism=str(
                    getattr(source, "mechanism", None)
                    or getattr(source, "__name__", type(source).__name__)
                ),
                authored_by=str(getattr(source, "authored_by", "unrecorded")),
                counters=counters,
            )
        )
    return RunTelemetry(
        mechanisms=tuple(mechanisms),
        real_evaluations=int(real_evaluations),
        virtual_evaluations=int(virtual_evaluations),
        proxy_evaluations=int(proxy_evaluations),
    )

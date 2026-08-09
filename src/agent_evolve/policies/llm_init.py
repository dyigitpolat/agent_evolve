"""Model-authored population initialization: the highest-leverage decision.

Initialization shapes the entire trajectory -- every later operator recombines
what the first generation contained. Until now that decision was schema-
uniform noise; this seam lets the model propose the initial population from
the domain card: on a sizing problem, textbook-informed starting points can
place the run in a different basin before a single evaluation is spent.

The authoring line, applied to initialization: every proposed member is
validated VALUE-BY-VALUE against the declared domains (declared loci must
hold declared values; undeclared loci must keep the template's value);
rejected members are counted and their slots fall back to schema-uniform
draws; the caller's own seeds always come first and always survive. The
budget then arbitrates: proposed members are evaluated like any other
candidate, and a bad initial population is paid for in the open.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent_evolve.policies.genetic import Locus, loci_of, locus_domain, read_locus

__all__ = ["InitTelemetry", "author_initial_population", "INIT_PROMPT"]

Config = Dict[str, Any]


@dataclass
class InitTelemetry:
    calls: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected_out_of_domain: int = 0
    rejected_shape: int = 0
    unparseable: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in
                ("calls", "proposed", "accepted", "rejected_out_of_domain",
                 "rejected_shape", "unparseable", "errors")}


INIT_PROMPT = """{context}

You are proposing the INITIAL POPULATION for an evolutionary search over the
problem above. These {k} configurations are the raw material every later
recombination works with; they should be individually strong bets AND
collectively diverse (covering distinct promising regions/trade-offs), based
on what the parameters and objectives MEAN.

Reply with ONLY a JSON list of exactly {k} configuration objects, each with
every parameter field, every value taken from that parameter's declared
domain. No commentary."""


def author_initial_population(
    complete: Callable[[str], str],
    *,
    candidate_model: Any,
    template: Config,
    k: int,
    domain_context: str,
    telemetry: Optional[InitTelemetry] = None,
) -> List[Config]:
    """Up to *k* validated initial members; whatever is rejected costs a slot,
    not the run -- the loop fills shortfalls schema-uniformly."""

    tel = telemetry if telemetry is not None else InitTelemetry()
    tel.calls += 1
    try:
        text = complete(INIT_PROMPT.format(context=domain_context, k=k))
    except Exception:
        tel.errors += 1
        return []
    match = re.search(r"\[.*\]", text, re.S)
    if match is None:
        tel.unparseable += 1
        return []
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        tel.unparseable += 1
        return []
    if not isinstance(raw, list):
        tel.unparseable += 1
        return []

    accepted: List[Config] = []
    for member in raw[:k]:
        tel.proposed += 1
        if not isinstance(member, dict) or set(member) != set(template):
            tel.rejected_shape += 1
            continue
        try:
            member_loci = loci_of(member)
        except Exception:
            tel.rejected_shape += 1
            continue
        if member_loci != loci_of(template):
            tel.rejected_shape += 1
            continue
        ok = True
        for locus in member_loci:
            value = read_locus(member, locus)
            domain = locus_domain(candidate_model, locus)
            if domain:
                if value not in domain:
                    ok = False
                    break
            elif value != read_locus(template, locus):
                ok = False
                break
        if not ok:
            tel.rejected_out_of_domain += 1
            continue
        tel.accepted += 1
        accepted.append(dict(member))
    return accepted

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

WHAT THE ASK IS FOR is a separate decision from who answers it, and ``style``
declares it. ``"joint"`` -- the default, and every sealed row -- asks for one
list that is individually strong AND collectively diverse. That single ask
conflates two things, and the ladder measured the cost: median pool regret is
better at reasoning effort ``none`` than at ``medium`` on 23 of 24 paired
seeds, because deliberation buys DIVERSITY and the pool median pays for it.
``"split"`` is the same one call with two labelled sub-asks -- k_exploit
strongest individual bets, k_explore configurations covering distinct regions
-- so effort and capability have a channel (the exploit subset) that is not
diversity-conflated, and the telemetry labels every returned member so that
subset can be scored on its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from agent_evolve.policies.genetic import Locus, loci_of, locus_domain, read_locus

__all__ = ["InitTelemetry", "author_initial_population", "INIT_PROMPT",
           "SPLIT_INIT_PROMPT", "INIT_STYLES"]

Config = Dict[str, Any]

#: What the one initialization call asks for. See the module docstring.
INIT_STYLES = ("joint", "split")

#: The labels a returned member can carry, and the counter prefixes that go
#: with them. ``"joint"`` and ``"unlabeled"`` are labels without a subset:
#: nothing in the reply distinguished those members, so nothing counts them
#: apart.
_SUBSETS = ("exploit", "explore")


@dataclass
class InitTelemetry:
    calls: int = 0
    proposed: int = 0
    accepted: int = 0
    rejected_out_of_domain: int = 0
    rejected_shape: int = 0
    unparseable: int = 0
    errors: int = 0
    #: The split ask's two halves, counted apart. Zero under ``"joint"``, and
    #: zero under a split reply that arrived as a bare list: a member nothing
    #: labelled is not evidence about either half.
    exploit_proposed: int = 0
    exploit_accepted: int = 0
    explore_proposed: int = 0
    explore_accepted: int = 0
    #: One label per RETURNED member, in the returned order -- so a scorer can
    #: read the exploit subset's quality without re-deriving which member came
    #: from which half. "exploit" | "explore" | "joint" | "unlabeled".
    labels: List[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in
                ("calls", "proposed", "accepted", "rejected_out_of_domain",
                 "rejected_shape", "unparseable", "errors",
                 "exploit_proposed", "exploit_accepted",
                 "explore_proposed", "explore_accepted")}


INIT_PROMPT = """{context}

You are proposing the INITIAL POPULATION for an evolutionary search over the
problem above. These {k} configurations are the raw material every later
recombination works with; they should be individually strong bets AND
collectively diverse (covering distinct promising regions/trade-offs), based
on what the parameters and objectives MEAN.

Reply with ONLY a JSON list of exactly {k} configuration objects, each with
every parameter field, every value taken from that parameter's declared
domain. No commentary."""


SPLIT_INIT_PROMPT = """{context}

You are proposing the INITIAL POPULATION for an evolutionary search over the
problem above. These {k} configurations are the raw material every later
recombination works with, and they are asked for in TWO parts. Each part is
optimized for its own thing and for nothing else.

EXPLOIT: exactly {k_exploit} configurations that are the STRONGEST INDIVIDUAL
BETS on the evidence of the card above -- what you would submit if only the
single best member counted. Do not spend any of these on coverage.

EXPLORE: exactly {k_explore} configurations covering DISTINCT promising
regions and trade-offs of the space -- what you would submit if only the
spread counted. Do not spend any of these on being individually safe.

Every configuration in both parts carries every parameter field, with every
value taken from that parameter's declared domain.

Reply with ONLY this JSON shape and no other text:
{{"exploit": [...], "explore": [...]}}"""


def _bare_list(text: str) -> Optional[list]:
    """The first bracketed run of *text* as a JSON list, or ``None``.

    The parse every sealed initialization ran, factored out unchanged so that
    both styles read a list the same way -- and so the split style's fallback
    reads exactly what a joint reply would have been read as.
    """

    match = re.search(r"\[.*\]", text, re.S)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return raw if isinstance(raw, list) else None


def _labelled_lists(text: str) -> Optional[tuple[list, list]]:
    """``(exploit, explore)`` from the split reply shape, or ``None``.

    ``None`` whenever either key is missing or is not a list: the reply did
    not answer the ask that was made, and this seam repairs nothing.
    """

    match = re.search(r"\{.*\}", text, re.S)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    exploit, explore = raw.get("exploit"), raw.get("explore")
    if not isinstance(exploit, list) or not isinstance(explore, list):
        return None
    return exploit, explore


def _rejection(member: Any, template: Config, candidate_model: Any) -> Optional[str]:
    """``None`` when *member* is admissible, else which counter it moves.

    Value-by-value against the declared domains: a declared locus must hold a
    declared value, an undeclared one must keep the template's. The one place
    a member is judged, so both styles are judged identically.
    """

    if not isinstance(member, dict) or set(member) != set(template):
        return "shape"
    try:
        member_loci = loci_of(member)
    except Exception:
        return "shape"
    if member_loci != loci_of(template):
        return "shape"
    for locus in member_loci:
        value = read_locus(member, locus)
        domain = locus_domain(candidate_model, locus)
        if domain:
            if value not in domain:
                return "domain"
        elif value != read_locus(template, locus):
            return "domain"
    return None


def author_initial_population(
    complete: Callable[[str], str],
    *,
    candidate_model: Any,
    template: Config,
    k: int,
    domain_context: str,
    telemetry: Optional[InitTelemetry] = None,
    style: str = "joint",
) -> List[Config]:
    """Up to *k* validated initial members; whatever is rejected costs a slot,
    not the run -- the loop fills shortfalls schema-uniformly.

    *style* is the ASK (see the module docstring). ``"joint"`` is the sealed
    one, prompt and parse unchanged. ``"split"`` is one call carrying two
    labelled sub-asks, ``k_exploit = (k + 1) // 2`` strongest bets first and
    the rest coverage; the halves are capped separately, members come back
    exploit-first, and every returned member is labelled in the telemetry. A
    split reply missing either key is read as a bare list if one is there --
    unlabelled, counted in no half -- and is otherwise unparseable. Nothing is
    repaired, in either style.
    """

    if style not in INIT_STYLES:
        raise ValueError(
            f"init style must be one of {INIT_STYLES}, got {style!r}")

    tel = telemetry if telemetry is not None else InitTelemetry()
    tel.calls += 1
    k_exploit = (k + 1) // 2
    prompt = (
        INIT_PROMPT.format(context=domain_context, k=k) if style == "joint"
        else SPLIT_INIT_PROMPT.format(
            context=domain_context, k=k, k_exploit=k_exploit,
            k_explore=k - k_exploit)
    )
    try:
        text = complete(prompt)
    except Exception:
        tel.errors += 1
        return []

    batches: tuple[tuple[str, list], ...]
    if style == "joint":
        raw = _bare_list(text)
        if raw is None:
            tel.unparseable += 1
            return []
        batches = (("joint", raw[:k]),)
    else:
        pair = _labelled_lists(text)
        if pair is not None:
            batches = (("exploit", pair[0][:k_exploit]),
                       ("explore", pair[1][:k - k_exploit]))
        else:
            raw = _bare_list(text)
            if raw is None:
                tel.unparseable += 1
                return []
            batches = (("unlabeled", raw[:k]),)

    accepted: List[Config] = []
    for label, members in batches:
        subset = label if label in _SUBSETS else None
        for member in members:
            tel.proposed += 1
            if subset:
                setattr(tel, f"{subset}_proposed",
                        getattr(tel, f"{subset}_proposed") + 1)
            reason = _rejection(member, template, candidate_model)
            if reason == "shape":
                tel.rejected_shape += 1
                continue
            if reason is not None:
                tel.rejected_out_of_domain += 1
                continue
            tel.accepted += 1
            if subset:
                setattr(tel, f"{subset}_accepted",
                        getattr(tel, f"{subset}_accepted") + 1)
            tel.labels.append(label)
            accepted.append(dict(member))
    return accepted

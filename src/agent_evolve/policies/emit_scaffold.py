"""The emit harness: a candidate's SHAPE is the harness's job, not the model's.

Wave D measured what happens when it is not. On the assignment-structured
`upms_j14_m3` instance the authored generator emitted 7,104 candidates and
the harness could use 23 of them (0.3%); **6,021 of the 7,081 rejects were
SHAPE** -- the key set, the nesting, the length -- and 1,060 were values
outside a locus's own declared domain. The same failure had already been
measured on `upms_j13_m3` (33.9% acceptance, 1,544 shape + 1,172
out-of-domain), and revision fired on 77 of 80 cells without repairing it.
Healthy instances on the same substrate run 67.3-96.0%.

Shape is the load-bearing half, and it is not a modelling problem: the
harness already knows the exact key set, the exact nesting and the exact
admissible value of every locus, because it derives all three from the
problem's own schema. Asking a model to reproduce that structure by hand --
fourteen keys named ``job_00``..``job_13``, each with its own eligibility
subset -- is asking it to re-derive something the caller could simply have
built, and the failure rate scales with the number of loci exactly as
transcription errors do.

So this module inverts the burden, generically, for ANY structured genome:

- :func:`scaffold_prelude` renders a source prelude the sandbox executes
  BEFORE the authored artifact. It defines ``build(picks)`` -- the authored
  code names loci and values, and the harness assembles the configuration.
  A configuration built through ``build`` cannot have the wrong shape,
  because the shape comes from the template rather than from the model, and
  cannot hold an out-of-domain value, because every value is checked against
  that locus's own domain before it is written. Both events are COUNTED and
  reported back through the runtime's notes channel, so "the model got it
  right" and "the harness repaired it" never look alike.
- :func:`coerce_candidate` is the same assembly, harness-side, for a
  generator that ignored the scaffold and hand-built its dicts anyway. It
  reads whatever loci a member does supply -- flat locus keys, the
  template's own nesting, or a bare sequence aligned with the loci -- and
  repairs the rest. A per-LOCUS fallback strictly dominates the per-
  CANDIDATE fallback it replaces: the run keeps the thirteen loci the model
  chose well instead of throwing all fourteen away because one was wrong.
- :func:`render_domain_echo` writes the per-locus admissible sets into the
  authoring prompt, so an out-of-domain value is a prompt failure rather
  than a guess. Sequence-valued fields are the case that motivates it: the
  domain card renders a sequence field's SHARED per-element vocabulary,
  while ``domains`` is keyed per position, and only the echo says which.

Nothing here knows what a locus means. The template, the loci and the
domains all arrive from :mod:`agent_evolve.policies.genetic`, which derives
them from the candidate model's own JSON schema.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.infrastructure.authored_worker import ALLOWED_IMPORTS
from agent_evolve.policies.genetic import Locus, loci_of, read_locus, write_locus

__all__ = [
    "NOTES_GLOBAL",
    "locus_names",
    "scaffold_prelude",
    "coerce_candidate",
    "render_domain_echo",
    "SCAFFOLD_RULES",
]

#: The name the prelude publishes its counters under. The worker returns
#: whatever lives here as the call's ``notes``; nothing else crosses back.
NOTES_GLOBAL = "_AE_EMIT_NOTES"

#: How many values of one locus the echo prints before eliding.
_MAX_VALUES = 12
#: How many locus lines the echo prints before collapsing the tail.
_MAX_LINES = 48
#: How many distinct offending samples a census keeps per reason.
_MAX_SAMPLES = 4


def locus_names(template: Mapping[str, Any]) -> Tuple[str, ...]:
    """Canonical locus names for *template*, in the harness's own order."""

    return tuple(str(locus) for locus in loci_of(template))


def _parse(name: str) -> Locus:
    """The inverse of ``str(Locus)``: ``field`` or ``field[i]``."""

    if name.endswith("]") and "[" in name:
        field, index = name[:-1].split("[", 1)
        try:
            return Locus(field, int(index))
        except ValueError:
            return Locus(name)
    return Locus(name)


# --------------------------------------------------------------- the prelude

_PRELUDE = '''\
{preimports}
import json as _ae_json
import random as _ae_random

_AE = _ae_json.loads({payload})
LOCI = tuple(_AE["loci"])
DOMAINS = {{k: list(v) for k, v in _AE["domains"].items()}}
_AE_TEMPLATE = _AE["template"]
{notes} = {{"built": 0, "filled": 0, "out_of_domain": 0, "unknown_locus": 0,
           "by_locus": {{}}, "samples": []}}
_AE_RNG = _ae_random.Random(_AE["nonce"])


def domain_of(locus):
    """The admissible values at *locus* -- the same list `domains` carries."""
    return list(DOMAINS.get(str(locus), ()))


def _ae_count(kind, locus, value=None):
    {notes}[kind] = {notes}.get(kind, 0) + 1
    row = {notes}["by_locus"].setdefault(str(locus), {{}})
    row[kind] = row.get(kind, 0) + 1
    if kind == "out_of_domain" and len({notes}["samples"]) < 8:
        try:
            _ae_json.dumps(value)
        except Exception:
            value = repr(value)[:80]
        {notes}["samples"].append({{"locus": str(locus), "value": value}})


def _ae_read(config, locus):
    if locus.endswith("]") and "[" in locus:
        field, index = locus[:-1].split("[", 1)
        return config[field][int(index)]
    return config[locus]


def _ae_write(config, locus, value):
    if locus.endswith("]") and "[" in locus:
        field, index = locus[:-1].split("[", 1)
        config[field][int(index)] = value
    else:
        config[locus] = value


def _ae_picks(picks):
    if picks is None:
        return {{}}
    if isinstance(picks, dict):
        flat = {{}}
        for key, value in picks.items():
            key = str(key)
            if key in DOMAINS or key in _AE_TEMPLATE:
                if (isinstance(value, (list, tuple))
                        and key not in DOMAINS
                        and isinstance(_AE_TEMPLATE.get(key), list)):
                    for i, item in enumerate(value):
                        flat["%s[%d]" % (key, i)] = item
                else:
                    flat[key] = value
            else:
                _ae_count("unknown_locus", key)
        return flat
    try:
        values = list(picks)
    except Exception:
        return {{}}
    return {{LOCI[i]: v for i, v in enumerate(values) if i < len(LOCI)}}


def build(picks=None, seed=None):
    """Assemble ONE configuration, locus by locus, from your choices.

    *picks* maps a locus name to the value you want there (or is a sequence
    aligned with ``LOCI``). The shape comes from the harness, so it is always
    right; a locus you omit, or set outside ``DOMAINS[locus]``, is filled by
    a seeded draw from that locus's own domain and counted against you.
    """
    rng = _ae_random.Random(seed) if seed is not None else _AE_RNG
    config = _ae_json.loads(_ae_json.dumps(_AE_TEMPLATE))
    chosen = _ae_picks(picks)
    for locus in LOCI:
        domain = DOMAINS.get(locus) or ()
        if locus in chosen:
            value = chosen[locus]
            if not domain:
                # The schema constrains this locus to nothing, so the template
                # value is the only defensible one; overwriting it would be
                # authoring a value the problem never declared.
                if value != _ae_read(config, locus):
                    _ae_count("out_of_domain", locus, value)
                continue
            if value in domain:
                _ae_write(config, locus, value)
                continue
            _ae_count("out_of_domain", locus, value)
        elif not domain:
            continue
        else:
            _ae_count("filled", locus)
        _ae_write(config, locus, rng.choice(domain))
    {notes}["built"] += 1
    return config


def build_all(rows, seed=None):
    """``build`` over an iterable of picks -- one configuration per row."""
    return [build(row, seed=seed) for row in rows]


def resample(locus, current=None, seed=None, rng=None):
    """A NEW value at *locus*, or the only one there is.

    The shipped mutation operator's exact rule. It exists because excluding
    the current value from a domain that holds one value leaves an empty
    sequence, and `random.choice([])` raises -- which is the single largest
    crash class measured in authored generators on heterogeneous genomes.
    """
    domain = list(DOMAINS.get(str(locus)) or ())
    if not domain:
        return current
    others = [v for v in domain if v != current] or domain
    source = rng if rng is not None else (
        _ae_random.Random(seed) if seed is not None else _AE_RNG)
    return source.choice(others)
'''


#: The paragraph the authoring prompt shows. Kept next to the code it
#: describes so the two cannot drift.
SCAFFOLD_RULES = """\
- Build EVERY configuration through the helper `build(picks)`. It is already
  defined in your module, as are the standard-library modules listed above --
  write `import random` if you like, but `random`, `math` and the rest are
  bound already, and `build` must not be redefined. Do NOT construct
  configuration dicts yourself. `picks` maps a locus name to the
  value you want there -- e.g. `build({"x[0]": 3, "mode": "fast"})` -- or is
  a sequence aligned with the module-level tuple `LOCI`. `build` returns a
  configuration of exactly the right shape; a locus you omit, or set to a
  value outside its domain, is filled by a uniform draw from that locus's
  own domain AND COUNTED AGAINST YOU, so omitting loci wastes the guidance
  you are being paid for. `build_all(rows)` does a whole batch.
- `LOCI` (tuple of locus names, in order), `DOMAINS` (the same mapping as
  the `domains` argument) and `domain_of(locus)` are available too.
- LOCI DO NOT SHARE A DOMAIN, and some hold a SINGLE admissible value. So
  `[v for v in domain_of(k) if v != current]` can be EMPTY, and
  `random.choice([])` raises -- which kills the whole batch and forfeits the
  pool. Use `resample(locus, current)` for "change this locus": it returns a
  different value where one exists and the only value where it does not."""


def _preimports() -> str:
    """Bind every ALLOWED module as a global before the artifact runs.

    A measured failure class, not a convenience. Replaying the DEV probe's
    zero-emission cells showed 18 of 51 crashes were `NameError: name
    'random'/'math' is not defined` -- an authored function using a module
    the gate already permits, without an import statement. The allowlist is
    the harness's own datum; binding it costs one import per module in a
    process that exists for one batch, and the gate is unchanged, because
    the gate is about what the ARTIFACT may import, not about what the
    harness may hand it.
    """

    return "\n".join(f"import {name}"
                     for name in sorted(ALLOWED_IMPORTS))


def scaffold_prelude(
    template: Mapping[str, Any],
    domains: Mapping[str, Sequence[Any]],
    *,
    nonce: int = 0,
) -> Optional[str]:
    """Source the sandbox runs before the artifact, or ``None`` if it cannot.

    ``None`` is returned when the template or a domain will not survive the
    JSON round trip the runtime already imposes on every argument -- in which
    case the artifact runs exactly as it did before this module existed and
    the harness-side guard is the only line of defence, which is the same
    degradation everything else here makes.
    """

    try:
        payload = json.dumps({
            "template": dict(template),
            "loci": list(locus_names(template)),
            "domains": {str(k): list(v) for k, v in domains.items()},
            "nonce": int(nonce),
        }, sort_keys=True)
    except (TypeError, ValueError):
        return None
    return _PRELUDE.format(payload=repr(payload), notes=NOTES_GLOBAL,
                           preimports=_preimports())


# ------------------------------------------------------- the harness-side twin

def _picks_from(
    member: Any,
    *,
    template: Mapping[str, Any],
    domains: Mapping[str, Sequence[Any]],
    loci: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Whatever loci *member* actually supplies, keyed by locus name.

    Three shapes are read, in the order a generator is likely to emit them:
    flat locus keys (what ``build`` produces), the template's own nesting
    (what a hand-built dict produces), and a bare sequence aligned with the
    loci (what an assignment-structured genome is naturally written as).
    ``None`` means the member supplied nothing addressable at all.
    """

    if isinstance(member, dict):
        flat: Dict[str, Any] = {}
        for key, value in member.items():
            key = str(key)
            if key in domains and not (
                    isinstance(value, (list, tuple))
                    and isinstance(template.get(key), (list, tuple))):
                flat[key] = value
            elif isinstance(value, (list, tuple)) and not isinstance(
                    value, (str, bytes)):
                for index, item in enumerate(value):
                    flat[f"{key}[{index}]"] = item
            elif key in template:
                flat[key] = value
        known = {k: v for k, v in flat.items() if k in domains}
        return known or None
    if isinstance(member, (list, tuple)) and not isinstance(member, (str, bytes)):
        if len(member) != len(loci):
            return None
        return dict(zip(loci, member))
    return None


def coerce_candidate(
    member: Any,
    *,
    template: Mapping[str, Any],
    domains: Mapping[str, Sequence[Any]],
    rng: random.Random,
    loci: Optional[Sequence[Locus]] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, List[str]]]:
    """Assemble a valid configuration out of whatever *member* got right.

    Returns ``(config, repairs)``; ``config`` is ``None`` when the member
    addressed no locus at all, which is the one case a per-locus fallback
    cannot improve on a rejection -- there is nothing of the model's left in
    it. ``repairs`` maps ``"filled"`` / ``"out_of_domain"`` to the loci the
    harness had to decide for.
    """

    positions = tuple(loci) if loci is not None else loci_of(template)
    names = [str(locus) for locus in positions]
    picks = _picks_from(member, template=template, domains=domains, loci=names)
    if picks is None:
        return None, {}

    repairs: Dict[str, List[str]] = {"filled": [], "out_of_domain": []}
    config: Dict[str, Any] = dict(template)
    kept = 0
    for locus, name in zip(positions, names):
        domain = tuple(domains.get(name) or ())
        if name in picks:
            value = picks[name]
            if not domain:
                if value != read_locus(template, locus):
                    repairs["out_of_domain"].append(name)
                continue
            if value in domain:
                config = write_locus(config, locus, value)
                kept += 1
                continue
            repairs["out_of_domain"].append(name)
        elif not domain:
            continue
        else:
            repairs["filled"].append(name)
        if domain:
            config = write_locus(config, locus, rng.choice(domain))
    if not kept:
        # Nothing the model chose survived. Accepting this would credit the
        # generator with a draw that is schema-uniform in every locus.
        return None, repairs
    return config, repairs


# ------------------------------------------------------------------ the echo

def _render_values(values: Sequence[Any]) -> str:
    values = list(values)
    if not values:
        return "unconstrained by the schema (keeps the template's value)"
    if len(values) <= _MAX_VALUES:
        return f"one of {values}"
    head = values[:_MAX_VALUES - 2]
    return f"one of {head + ['...', values[-1]]} ({len(values)} values)"


def render_domain_echo(
    domains: Mapping[str, Sequence[Any]],
    *,
    max_lines: int = _MAX_LINES,
) -> str:
    """Every locus's exact admissible set, one line each, runs collapsed.

    Consecutive loci that share a domain -- the common case for a sequence
    field, and the case the domain card already renders correctly -- collapse
    to one line naming the run, so a 500-position genome costs one line while
    a 14-position genome whose positions differ (per-job eligibility, per-slot
    capability) spends fourteen and says exactly which is which. That
    difference is the whole point: an out-of-domain value must be a prompt
    failure, not a guess.
    """

    names = list(domains)
    if not names:
        return ""
    runs: List[Tuple[List[str], Sequence[Any]]] = []
    for name in names:
        values = list(domains[name] or ())
        if runs and list(runs[-1][1]) == values:
            runs[-1][0].append(name)
        else:
            runs.append(([name], values))

    lines: List[str] = []
    for group, values in runs[:max_lines]:
        label = group[0] if len(group) == 1 else f"{group[0]} .. {group[-1]}"
        if len(group) > 1:
            label += f" ({len(group)} loci)"
        lines.append(f"  {label}: {_render_values(values)}")
    if len(runs) > max_lines:
        remaining = sum(len(group) for group, _ in runs[max_lines:])
        lines.append(f"  ... and {remaining} further loci; the `domains` "
                     f"argument carries all of them at call time.")
    singletons = [name for name in names if len(list(domains[name] or ())) == 1]
    if singletons:
        # Called out because it is the shape of a measured crash, not because
        # it is interesting: a locus with one admissible value makes
        # "pick something else here" an empty draw.
        shown = singletons[:8]
        tail = ("" if len(singletons) <= 8
                else f" and {len(singletons) - 8} more")
        lines.append(f"  NOTE: {len(singletons)} locus/loci admit exactly ONE "
                     f"value ({', '.join(shown)}{tail}) -- they cannot be "
                     f"changed, and excluding their current value leaves an "
                     f"empty domain.")
    return "\n".join(lines)

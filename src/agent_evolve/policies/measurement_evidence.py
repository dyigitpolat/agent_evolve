"""What the run MEASURED, rendered so a model can reason about it.

Every authored mechanism in this package so far has been conditioned on
MEANING: the model reads a semantic card and a schema and writes an artifact
before a single evaluation exists. That is a static prior. It is worth a great
deal where a recallable domain prior is strong and it is worth exactly nothing
where one is not, and no amount of re-prompting changes which of those two a
venue is -- because the prompt never contains a measurement.

This module is the other half: it turns the run's own measured
``(configuration -> objectives)`` trace into evidence a model can be asked to
reason over. It is deliberately a pure function of the trace:

- it never sees the problem, the evaluator, the cache or the budget, so
  rendering evidence cannot spend one;
- what it emits is TEXT plus a digest of that text, so a run can record
  exactly what the model was shown and no run can imply it reasoned over
  measurements when it did not;
- it computes nothing a reader cannot recompute from the same rows.

Three things are rendered, in the order the evidence supports them:

``front``       the non-dominated members measured so far -- the run's own
                weight-free verdict on what is good, the same relation
                selection already uses.
``progress``    per objective, the best value in the first half of the trace
                against the best in the second, so "what improved and what
                did not" is a measurement and not an adjective.
``effects``     per (locus, objective) rank correlation over the measured
                rows: WHICH KNOB MOVES WHICH COST. This is the term a genetic
                algorithm can only discover by stumbling and a reader of
                twenty measured points should be able to name.

``coverage`` rides along with ``effects``: which declared values a locus has
actually been measured at, so "this region is exhausted" is checkable.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import dominates
from agent_evolve.policies.genetic import Locus, loci_of, read_locus

__all__ = [
    "MeasuredRow",
    "EvidenceView",
    "front_of",
    "spearman",
    "locus_effects",
    "render_measurement_evidence",
    "evidence_digest",
    "LocusPrior",
    "PriorVerdict",
    "parse_locus_prior",
    "admit_locus_prior",
    "LOCUS_PRIOR_PROMPT",
]

#: One measured candidate: its configuration, what the evaluator returned,
#: and whether it survived selection. Survival is the run's own weight-free
#: quality verdict and costs nothing extra to carry.
MeasuredRow = Tuple[Mapping[str, Any], Mapping[str, float], bool]

#: The seam a CONTROL arm wraps. A view receives the rows this run measured
#: and returns the rows the model will be shown. The identity view is the
#: product; a view that returns another run's rows -- same count, same shape,
#: same cost, wrong run -- is the shuffled-evidence control that says whether
#: any measured gain comes from reasoning over THIS run's evidence or merely
#: from making another call. It lives here, as a declared parameter, because a
#: mechanism whose control cannot be built without editing the product is a
#: mechanism whose control will not be built.
EvidenceView = Callable[[Sequence[MeasuredRow]], Sequence[MeasuredRow]]


def front_of(
    rows: Sequence[MeasuredRow],
    specs: Sequence[ObjectiveSpec],
) -> List[MeasuredRow]:
    """The non-dominated subset of *rows*, in measurement order."""

    kept: List[MeasuredRow] = []
    for row in rows:
        if any(dominates(dict(other[1]), dict(row[1]), specs)
               for other in rows if other is not row):
            continue
        kept.append(row)
    return kept


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not manufacture correlation."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rank correlation, or ``None`` when it is not defined.

    Undefined means undefined and is reported as such: fewer than three
    pairs, or a constant column, cannot produce a correlation and must not
    produce a zero that reads like "measured, no effect".
    """

    if len(xs) != len(ys) or len(xs) < 3:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(list(xs)), _ranks(list(ys))
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    if den <= 0.0:
        return None
    return max(-1.0, min(1.0, num / den))


@dataclass(frozen=True)
class LocusEffect:
    """One locus, and what the trace says it does to each objective."""

    locus: str
    #: objective name -> rank correlation between the locus's position in its
    #: DECLARED value order and that objective. ``None`` where undefined.
    correlation: Mapping[str, Optional[float]]
    #: declared value -> how many measured rows hold it.
    coverage: Mapping[str, int]
    unmeasured: Tuple[str, ...]

    @property
    def strength(self) -> float:
        return max((abs(v) for v in self.correlation.values() if v is not None),
                   default=0.0)


def locus_effects(
    rows: Sequence[MeasuredRow],
    specs: Sequence[ObjectiveSpec],
    domains: Mapping[str, Sequence[Any]],
) -> List[LocusEffect]:
    """Per-locus rank correlations and coverage, strongest effect first.

    The predictor is a locus's POSITION IN ITS DECLARED VALUE ORDER, not its
    value: the declared order is the only ordering this package is allowed to
    assume about a categorical domain, it is what the schema published, and it
    is the same order the sampler draws from. Where a domain is genuinely
    unordered the correlation is meaningless and the model is told the
    coverage instead -- which is why both travel together.
    """

    if not rows:
        return []
    template = dict(rows[0][0])
    effects: List[LocusEffect] = []
    for locus in loci_of(template):
        key = str(locus)
        domain = list(domains.get(key) or ())
        if len(domain) < 2:
            continue
        index_of = {_token(v): i for i, v in enumerate(domain)}
        positions: List[float] = []
        keep: List[int] = []
        counts: Dict[str, int] = {_token(v): 0 for v in domain}
        for i, (config, _obj, _s) in enumerate(rows):
            try:
                token = _token(read_locus(config, locus))
            except Exception:
                continue
            if token not in index_of:
                continue
            counts[token] += 1
            positions.append(float(index_of[token]))
            keep.append(i)
        correlation: Dict[str, Optional[float]] = {}
        for spec in specs:
            column = [float(rows[i][1].get(spec.name, 0.0)) for i in keep]
            correlation[spec.name] = spearman(positions, column)
        effects.append(LocusEffect(
            locus=key,
            correlation=correlation,
            coverage=dict(counts),
            unmeasured=tuple(t for t, c in counts.items() if c == 0),
        ))
    effects.sort(key=lambda e: (-e.strength, e.locus))
    return effects


def _token(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _fmt(value: float) -> str:
    return f"{float(value):.6g}"


def render_measurement_evidence(
    rows: Sequence[MeasuredRow],
    specs: Sequence[ObjectiveSpec],
    domains: Mapping[str, Sequence[Any]],
    *,
    front_shown: int = 8,
    effects_shown: int = 8,
    charged: Optional[int] = None,
) -> str:
    """The measured trace as prompt text. Pure, and a pure function of *rows*.

    Nothing here is an opinion. Every line is a count, a measured value or a
    rank correlation over the rows passed in, so the same rows always render
    the same bytes and :func:`evidence_digest` of that text identifies the
    evidence a call was conditioned on.
    """

    rows = list(rows)
    if not rows:
        return "  no candidate has been measured yet."
    lines: List[str] = [
        f"  measured so far: {len(rows)} configurations"
        + (f" ({charged} charged evaluations)" if charged is not None else ""),
    ]

    front = front_of(rows, specs)
    lines.append(f"  the current non-dominated front ({len(front)} of "
                 f"{len(rows)} measured):")
    for config, objectives, _s in front[:front_shown]:
        rendered = ", ".join(f"{s.name}={_fmt(objectives.get(s.name, 0.0))}"
                             for s in specs)
        lines.append(f"    {json.dumps(config, sort_keys=True, default=str)}"
                     f"  ->  {rendered}")
    if len(front) > front_shown:
        lines.append(f"    ... and {len(front) - front_shown} more front members")

    lines.append("  progress -- best value per objective, first half of the "
                 "run vs second half:")
    half = max(1, len(rows) // 2)
    for spec in specs:
        early = [float(o.get(spec.name, 0.0)) for _c, o, _s in rows[:half]]
        late = [float(o.get(spec.name, 0.0)) for _c, o, _s in rows[half:]]
        pick = min if spec.goal == "min" else max
        best_early = pick(early) if early else float("nan")
        best_late = pick(late) if late else best_early
        moved = ("IMPROVED" if late and pick([best_early, best_late]) == best_late
                 and best_late != best_early else "did not improve")
        lines.append(f"    {spec.name} ({spec.goal}imise): {_fmt(best_early)} "
                     f"-> {_fmt(best_late)}  [{moved}]")

    effects = locus_effects(rows, specs, domains)
    if effects:
        lines.append("  which parameter moves which cost -- rank correlation "
                     "between the parameter's position in its declared value "
                     "list and the measured cost (+1 = later values cost more, "
                     "-1 = later values cost less, '-' = not determinable "
                     "from these rows):")
        for effect in effects[:effects_shown]:
            body = ", ".join(
                f"{name}={'-' if value is None else f'{value:+.2f}'}"
                for name, value in effect.correlation.items())
            unmeasured = (f"; never measured at: "
                          f"{', '.join(effect.unmeasured)}"
                          if effect.unmeasured else "; every declared value "
                          "has been measured")
            lines.append(f"    {effect.locus}: {body}{unmeasured}")
        weak = [e.locus for e in effects[effects_shown:] if e.strength < 0.2]
        if weak:
            lines.append(f"    (no determinable effect on any cost: "
                         f"{', '.join(weak[:12])})")
    return "\n".join(lines)


def evidence_digest(text: str) -> str:
    """The identity of one evidence rendering. Recorded on every call."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ===================================================== the locus-importance channel

LOCUS_PRIOR_PROMPT = """You are advising a black-box multi-objective \
optimizer about WHERE IN ITS SEARCH SPACE to concentrate the rest of its
evaluations.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE -- every parameter and the values it may take:
{domains}

WHAT THE OPTIMIZER HAS ACTUALLY MEASURED SO FAR:

{evidence}

Read the measurements, not the parameter names. Decide which parameters
actually move the measured costs, and for those parameters which values are
worth spending the remaining budget on.

Reply with ONLY a JSON object, no prose and no code fence, of this shape:

{{"restrict": {{"<parameter>": ["<value>", ...], ...}},
 "because": {{"<parameter>": "<one short sentence citing the measurements>"}}}}

Rules, and the harness checks every one of them:
- Name ONLY parameters that appear in the search space above, and ONLY values
  that parameter declares. Anything else and the whole reply is REFUSED.
- You may only NARROW. Every value you drop is a value the optimizer will not
  sample again this run, so a parameter you cannot justify from the
  measurements should simply be left out.
- Never drop a value that a current non-dominated front member holds. The
  harness refuses the whole reply if you do -- excluding a measured optimum is
  the one failure that cannot be recovered from.
- Leave "restrict" empty if the measurements do not justify narrowing
  anything. An honest empty answer is accepted and costs nothing."""


@dataclass(frozen=True)
class LocusPrior:
    """A model-authored narrowing of declared domains, with its reasons."""

    restrict: Mapping[str, Tuple[Any, ...]] = ()
    because: Mapping[str, str] = ()

    def as_note(self) -> Dict[str, Any]:
        return {"restrict": {k: list(v) for k, v in dict(self.restrict).items()},
                "because": dict(self.because)}


@dataclass(frozen=True)
class PriorVerdict:
    """Admitted or refused, and WHY. The gate refuses; it never trusts."""

    admitted: bool
    reason: str = ""
    prior: Optional[LocusPrior] = None
    #: log10 of the factor the surviving space is smaller by.
    narrowing: float = 0.0

    def as_note(self) -> Dict[str, Any]:
        note: Dict[str, Any] = {"admitted": bool(self.admitted),
                                "reason": self.reason,
                                "narrowing_log10": round(self.narrowing, 4)}
        if self.prior is not None:
            note.update(self.prior.as_note())
        return note


def parse_locus_prior(text: str) -> Optional[LocusPrior]:
    """The model's reply as a typed prior, or ``None`` if it is not one.

    Whole-reply acceptance, exactly as the authored-artifact gate does it: a
    reply that is not a JSON object of the declared shape is not repaired into
    one, because a repaired prior is a prior nobody authored.
    """

    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1] if body.count("```") >= 2 else body
        if body.startswith("json"):
            body = body[4:]
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(body[start:end + 1])
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    restrict = parsed.get("restrict")
    if restrict is None:
        restrict = {}
    if not isinstance(restrict, dict):
        return None
    typed: Dict[str, Tuple[Any, ...]] = {}
    for name, values in restrict.items():
        if not isinstance(values, list):
            return None
        typed[str(name)] = tuple(values)
    because = parsed.get("because") or {}
    if not isinstance(because, dict):
        because = {}
    return LocusPrior(restrict=typed,
                      because={str(k): str(v) for k, v in because.items()})


def admit_locus_prior(
    prior: Optional[LocusPrior],
    *,
    domains: Mapping[str, Sequence[Any]],
    front: Sequence[MeasuredRow] = (),
    max_narrowing_log10: float = 2.0,
    min_values_kept: int = 1,
) -> PriorVerdict:
    """Refuse, or admit with the narrowing measured. Never repair.

    Five refusals, in the order a wrong prior does damage:

    ``unparsed``      there is no typed prior to judge.
    ``empty``         it narrows nothing, so there is nothing to measure and
                      nothing to admit -- an honest no-op, recorded as one.
    ``undeclared``    a parameter or a value the schema never declared. The
                      whole reply goes, not the offending entry: a prior that
                      is wrong about the schema has not read the schema, and
                      dropping only the bad line would admit the rest on the
                      strength of an author that demonstrably guessed.
    ``excludes_front``  it drops a value held by a MEASURED non-dominated
                      member. This is the catastrophe the channel exists to
                      be safe against -- a restriction that excludes the true
                      optimum cannot be recovered from by any later
                      generation -- so it is checked against the run's own
                      measurements rather than argued about.
    ``over_narrow``   it shrinks the space by more than
                      ``max_narrowing_log10`` orders of magnitude, or empties
                      a parameter below ``min_values_kept``. Concentration is
                      the point; collapse is not.
    """

    if prior is None:
        return PriorVerdict(False, "unparsed")
    restrict = {k: tuple(v) for k, v in dict(prior.restrict).items() if v is not None}
    if not restrict:
        return PriorVerdict(False, "empty", prior)

    declared = {str(k): [_token(v) for v in (values or ())]
                for k, values in dict(domains).items()}
    narrowing = 0.0
    for name, values in restrict.items():
        if name not in declared or not declared[name]:
            return PriorVerdict(False, f"undeclared parameter {name!r}", prior)
        allowed = set(declared[name])
        tokens = [_token(v) for v in values]
        if not tokens or any(t not in allowed for t in tokens):
            return PriorVerdict(False, f"undeclared value for {name!r}", prior)
        kept = len(set(tokens))
        if kept < min_values_kept:
            return PriorVerdict(False, f"{name!r} kept {kept} values", prior)
        if kept > len(allowed):                      # cannot happen; cheap guard
            return PriorVerdict(False, f"{name!r} widens its domain", prior)
        narrowing += math.log10(len(allowed) / kept) if kept else 0.0

    if narrowing <= 0.0:
        return PriorVerdict(False, "empty", prior, narrowing)
    for config, _objectives, _s in front:
        for name, values in restrict.items():
            tokens = {_token(v) for v in values}
            for locus in loci_of(dict(config)):
                if str(locus) != name:
                    continue
                if _token(read_locus(config, locus)) not in tokens:
                    return PriorVerdict(
                        False,
                        f"excludes a measured front member at {name!r}",
                        prior, narrowing)
    if narrowing > max_narrowing_log10:
        return PriorVerdict(False, f"over_narrow ({narrowing:.2f} decades)",
                            prior, narrowing)
    return PriorVerdict(True, "admitted", prior, narrowing)


def apply_locus_prior(
    domains: Mapping[str, Sequence[Any]],
    prior: LocusPrior,
) -> Dict[str, List[Any]]:
    """*domains* narrowed by an ADMITTED *prior*. Narrows only, never empties."""

    out: Dict[str, List[Any]] = {k: list(v) for k, v in dict(domains).items()}
    for name, values in dict(prior.restrict).items():
        tokens = {_token(v) for v in values}
        kept = [v for v in out.get(name, ()) if _token(v) in tokens]
        if kept:
            out[name] = kept
    return out

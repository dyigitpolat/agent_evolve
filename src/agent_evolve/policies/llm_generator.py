"""The model writes the SAMPLER, not the samples.

Every guidance mechanism before this one paid per decision: a chooser picks
parents once per offspring, an immigration call authors a handful of members
once per segment. Leverage like that scales as 1/budget, which is exactly why
per-decision guidance washes out on cheap-evaluation venues -- at B=40 one
call shapes a tenth of the run, at B=40,000 it shapes a ten-thousandth.

An authored GENERATOR inverts that economics. The model is asked once, before
any evaluation, to write ``propose(archive, n, domains, seed)`` -- a
distribution, not a draw -- and that one fixed cost then shapes every
candidate the run ever considers. Its quality is directly measurable and was
already measured before this seam existed (W3's induced-sampler recall
0.094/0.211/0.205 against 0.0585 chance).

The authoring line holds by the same construction as everywhere else, and it
is worth stating precisely because a generator emits candidates:

- Every emitted candidate is validated VALUE-BY-VALUE against the declared
  domains and the template's shape (``validate_pool``); rejects are counted
  and their slots fall back to schema-uniform draws, so a broken generator
  degrades to the credential-free sampler rather than to nothing.
- The emitted set is measured for COLLAPSE -- duplicates inside the batch and
  candidates already measured earlier in the run -- because a generator that
  returns the same configuration n times is not a sampler, and the difference
  has to be visible in the telemetry rather than inferred from a flat curve.
- This module is starved exactly as ``session.screening`` is: it receives a
  template, a candidate model, a restriction and an archive of configurations
  -- never the problem, never the evaluation cache. Mass generation therefore
  cannot spend budget: the only route from a generated candidate to a real
  evaluation is the loop measuring the ``want`` of them it could already
  afford, which is a property of the import graph rather than a convention.
- It EVOLVES with feedback: the revision hook shows the generator its own
  source plus what the harness measured about its output -- acceptance rate,
  duplicate and archive-overlap rates, and how many of its candidates
  survived selection -- and asks for a rewrite under the identical gate.

What the seam does NOT ask the model to do, since Wave D measured what
happens when it does. On an assignment-structured genome (`upms_j14_m3`,
fourteen scalar loci with per-locus eligibility) the sealed generator emitted
7,104 candidates against 39,993 uniformly-filled pool slots and 23
acceptances; `upms_j13_m3` reproduced it. The failure decomposes into three
things the HARNESS already knows and the model was left to re-derive:

- the SHAPE. ``policies.emit_scaffold`` ships a ``build(picks)`` helper into
  the sandbox, so authored code names loci and values and the harness
  assembles the configuration -- and assembles a partially-correct emission
  rather than dropping it whole, a per-LOCUS fallback in place of a
  per-CANDIDATE one.
- the DOMAINS. Every locus's admissible set is echoed into the authoring
  prompt (``render_domain_echo``), because a field-level card cannot say
  what a per-position domain is.
- the RESOURCE BUDGET. A batch that overruns the sandbox returns nothing, so
  its whole pool falls back to uniform -- invisible in a per-candidate
  counter. The wall/CPU/memory contract is echoed too, and one overrun buys
  a retry at ``n // 4`` rather than the loss of the pool.

Each is counted separately, ablatable separately, and off restores the
sealed behaviour exactly.

Two of those channels fire on EMISSION DEFECTS -- a rejected candidate, a
collapsed batch, a generator whose children never survive. That is a repair
loop for a broken sampler, and it is not the same thing as guidance: a
generator that emits perfectly valid candidates from a region the run has
already measured to be bad is never revised at all, because nothing about it
is defective. The prior it encodes is STATIC -- written once from the semantic
card, in a ladder cell from an empty archive -- and where a recallable domain
prior is strong that is worth a great deal, while where one is not it is worth
exactly nothing, three times measured.

``reauthor_every`` is the other trigger, and it fires on SEARCH PROGRESS
rather than on defects: after the run has charged that many evaluations, the
generator is re-authored against the measured
``(configuration -> objectives)`` trace -- the current front, what improved
and what did not, and which parameter the measurements say moves which cost
(:mod:`agent_evolve.policies.measurement_evidence`). ``locus_prior`` is the
second consumer of the same evidence: the model names the parameters and
values worth the remaining budget, the harness types the answer as a
narrowing of the declared domains, and a GATE refuses it -- rather than
trusts it -- whenever it is undeclared, empty, over-narrow, or excludes a
value a measured non-dominated member holds.

Both are OFF by default (``reauthor_every=0``), and off is the seam that ran
every sealed row to date. Both record what evidence the model was shown (its
digest), what it emitted, and whether the emission was accepted, so no run can
imply it reasoned over measurements when it did not.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.authored import CONTRACTS, AuthoredArtifact, authored_artifact
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.authored_worker import ALLOWED_IMPORTS
from agent_evolve.policies.emit_scaffold import (
    NOTES_GLOBAL,
    SCAFFOLD_RULES,
    coerce_candidate,
    render_domain_echo,
    scaffold_prelude,
)
from agent_evolve.policies.genetic import (
    loci_of,
    locus_domain,
    read_locus,
    uniform_candidate,
)
from agent_evolve.policies.llm_surrogate import (
    AuthorTelemetry,
    accept_block,
    json_compact,
)
from agent_evolve.policies.measurement_evidence import (
    LOCUS_PRIOR_PROMPT,
    MeasuredRow,
    admit_locus_prior,
    apply_locus_prior,
    evidence_digest,
    front_of,
    parse_locus_prior,
    render_measurement_evidence,
)

__all__ = [
    "GeneratorTelemetry",
    "PoolReport",
    "RejectionCensus",
    "AuthoredGenerator",
    "author_generator",
    "revise_generator",
    "reauthor_generator",
    "render_generation_feedback",
    "validate_pool",
    "candidate_key",
    "GENERATOR_PROMPT",
    "GENERATOR_REVISION_PROMPT",
    "GENERATOR_EVIDENCE_PROMPT",
]

Config = Dict[str, Any]

#: How many archive members a prompt-side call carries into the sandbox. The
#: contract says "the archive"; shipping all of it would make a 10,000-
#: evaluation run pay a growing JSON round trip every generation for
#: information the sampler cannot use anyway.
ARCHIVE_SHOWN = 32

#: A hard ceiling on one batch, so "a configured pool size up to very large"
#: cannot become an accidental out-of-memory. Mass generation is the point;
#: an unbounded request is not.
MAX_POOL = 200_000


def candidate_key(config: Mapping[str, Any]) -> str:
    """Identity of a configuration, matching the loop's own dedup key."""

    return json.dumps(config, sort_keys=True, default=str)


@dataclass
class GeneratorTelemetry:
    """What mass generation actually did. Counted, never inferred.

    Rates are deliberately absent: :func:`~agent_evolve.core.telemetry.
    harvest_telemetry` coerces counters to ``int``, so every rate this
    mechanism claims is published as its exact numerator and denominator
    (``duplicates`` over ``emitted``) and computed by whoever reads them.
    :class:`PoolReport` carries the same ratios as floats for callers in
    process.
    """

    batches: int = 0
    runtime_failures: int = 0
    emitted: int = 0
    accepted: int = 0
    rejected_shape: int = 0
    rejected_out_of_domain: int = 0
    duplicates: int = 0
    archive_overlap: int = 0
    filled_uniform: int = 0
    measured: int = 0
    survived: int = 0
    revisions: int = 0
    revisions_accepted: int = 0
    #: Candidates the harness ASSEMBLED rather than rejected: the emitted
    #: member addressed at least one locus with an admissible value, and the
    #: rest of the configuration was filled from the template and the
    #: domains. Counted apart from ``accepted`` because a repaired candidate
    #: carries less of the model's guidance than a clean one, and a mechanism
    #: that only works after repair must not read as one that works.
    repaired: int = 0
    #: Individual loci the harness had to decide inside those candidates.
    repaired_loci: int = 0
    #: What the in-sandbox emit scaffold reported about its own work: loci
    #: the authored code left unset, and values it asked for that were not in
    #: that locus's domain. Both are counted at the point of construction, so
    #: they are visible even when nothing is rejected at all.
    scaffold_filled: int = 0
    scaffold_out_of_domain: int = 0
    #: Locus names the authored code used that the schema does not declare.
    scaffold_unknown_locus: int = 0
    #: Revisions that were authored, ran, and did NOT improve the measured
    #: defect -- the population the rejected-edit memory is built from.
    revisions_rejected: int = 0
    #: Batches that blew the sandbox's wall/CPU/memory budget and were retried
    #: at a smaller ``n``, and how many of those retries came back usable.
    runtime_retries: int = 0
    runtime_recovered: int = 0
    #: The MEASUREMENT-CONDITIONED channel, counted apart from the
    #: defect-triggered one above, because they are different mechanisms and a
    #: campaign that cannot tell them apart cannot attribute anything.
    #: ``reauthorings`` fired; ``reauthorings_accepted`` came back as a usable
    #: artifact; ``evidence_rows_shown`` is how many measured rows the model
    #: was actually given across those calls.
    reauthorings: int = 0
    reauthorings_accepted: int = 0
    evidence_rows_shown: int = 0
    #: The locus-importance channel. ``priors_refused`` is the number the GATE
    #: threw out and is the counter that says the gate is doing its job;
    #: ``priors_unwound`` counts admitted priors later dropped for not paying.
    priors_proposed: int = 0
    priors_admitted: int = 0
    priors_refused: int = 0
    priors_unwound: int = 0

    def as_dict(self) -> dict[str, int]:
        return {f.name: int(getattr(self, f.name)) for f in fields(self)}


@dataclass
class RejectionCensus:
    """WHICH loci rejected, WHY, and one concrete offending sample of each.

    Wave D's counters said 6,021 shape and 1,060 out-of-domain and could say
    nothing more, so the revision prompt could only tell the model that
    something was wrong -- which is why revision fired on 77 of 80 cells and
    repaired none of them. A revision is a repair instruction, and a repair
    instruction needs the address of the fault: the locus, the reason, and a
    value the model can recognise as its own.

    The routing is the point (program section 9-B5, the SHE borrowing):
    a defect is diagnosed against the artifact responsible for it, not
    aggregated into a rate that names nobody.
    """

    shape_reasons: Dict[str, int] = None            # type: ignore[assignment]
    out_of_domain_by_locus: Dict[str, int] = None   # type: ignore[assignment]
    repaired_by_locus: Dict[str, int] = None        # type: ignore[assignment]
    samples: Dict[str, Any] = None                  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("shape_reasons", "out_of_domain_by_locus",
                     "repaired_by_locus", "samples"):
            if getattr(self, name) is None:
                setattr(self, name, {})

    def shape(self, reason: str, sample: Any = None) -> None:
        self.shape_reasons[reason] = self.shape_reasons.get(reason, 0) + 1
        self.samples.setdefault(f"shape:{reason}", _sample_of(sample))

    def out_of_domain(self, locus: str, value: Any) -> None:
        self.out_of_domain_by_locus[locus] = (
            self.out_of_domain_by_locus.get(locus, 0) + 1)
        self.sample(locus, value)

    def sample(self, locus: str, value: Any) -> None:
        """One concrete offending value at *locus*; the first one sticks."""

        self.samples.setdefault(f"domain:{locus}", _sample_of(value))

    def repaired(self, locus: str) -> None:
        self.repaired_by_locus[locus] = self.repaired_by_locus.get(locus, 0) + 1

    def merge(self, other: "RejectionCensus") -> None:
        for reason, count in other.shape_reasons.items():
            self.shape_reasons[reason] = self.shape_reasons.get(reason, 0) + count
        for locus, count in other.out_of_domain_by_locus.items():
            self.out_of_domain_by_locus[locus] = (
                self.out_of_domain_by_locus.get(locus, 0) + count)
        for locus, count in other.repaired_by_locus.items():
            self.repaired_by_locus[locus] = (
                self.repaired_by_locus.get(locus, 0) + count)
        for key, value in other.samples.items():
            self.samples.setdefault(key, value)

    @property
    def empty(self) -> bool:
        return not (self.shape_reasons or self.out_of_domain_by_locus
                    or self.repaired_by_locus)

    def signature(self) -> str:
        """A stable name for THIS defect, so a repeat is recognisable.

        Two revisions that leave the same loci failing for the same reasons
        have not changed anything the harness can measure, whatever else they
        changed -- and that is exactly what the rejected-edit memory must be
        able to say back to the next revision.
        """

        parts = ([f"shape:{k}" for k in sorted(self.shape_reasons)]
                 + [f"domain:{k}" for k in sorted(self.out_of_domain_by_locus)])
        return "|".join(parts) or "clean"


def _sample_of(value: Any) -> Any:
    """A JSON-safe, bounded rendering of one offending value."""

    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)[:120]
    if isinstance(value, str) and len(value) > 120:
        return value[:120]
    return value


@dataclass(frozen=True)
class PoolReport:
    """One mass-generation batch, as the harness received it."""

    accepted: Tuple[Config, ...] = ()
    emitted: int = 0
    rejected_shape: int = 0
    rejected_out_of_domain: int = 0
    duplicates: int = 0
    archive_overlap: int = 0
    #: Accepted by ASSEMBLY rather than as emitted (see ``repair`` below).
    repaired: int = 0
    repaired_loci: int = 0
    census: RejectionCensus = None                  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.census is None:
            object.__setattr__(self, "census", RejectionCensus())

    def _rate(self, count: int) -> float:
        return (count / self.emitted) if self.emitted else 0.0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of the emitted batch the VALIDATION let through.

        Distinct from :attr:`novelty_rate`: a generator can be perfectly
        in-domain and still emit one configuration a thousand times, and the
        two guards have to be readable apart.
        """

        return self._rate(self.emitted - self.rejected_shape
                          - self.rejected_out_of_domain)

    @property
    def duplicate_rate(self) -> float:
        """Fraction of the emitted batch that repeated an earlier member."""

        return self._rate(self.duplicates)

    @property
    def archive_overlap_rate(self) -> float:
        """Fraction of the emitted batch already measured in this run."""

        return self._rate(self.archive_overlap)

    @property
    def novelty_rate(self) -> float:
        """Fraction of the emitted batch that was both valid and new."""

        return self._rate(len(self.accepted))

    @property
    def defect_rate(self) -> float:
        """Fraction of the batch the harness had to reject OR assemble.

        The one number a revision must move. Repairs are counted as defects
        here even though they were accepted: a candidate the harness had to
        finish is a candidate the model did not write, and a revision that
        turns rejections into repairs has moved the failure rather than
        fixed it.
        """

        return self._rate(self.rejected_shape + self.rejected_out_of_domain
                          + self.repaired)

    def as_note(self) -> Dict[str, Any]:
        """The per-generation history record: counts plus the guard's rates."""

        return {
            "emitted": self.emitted,
            "accepted": len(self.accepted),
            "rejected_shape": self.rejected_shape,
            "rejected_out_of_domain": self.rejected_out_of_domain,
            "duplicates": self.duplicates,
            "archive_overlap": self.archive_overlap,
            "repaired": self.repaired,
            "repaired_loci": self.repaired_loci,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "duplicate_rate": round(self.duplicate_rate, 4),
            "archive_overlap_rate": round(self.archive_overlap_rate, 4),
            "novelty_rate": round(self.novelty_rate, 4),
            "defect_rate": round(self.defect_rate, 4),
        }


def validate_pool(
    emitted: Any,
    *,
    template: Config,
    domains: Mapping[str, Sequence[Any]],
    seen: Optional[Any] = None,
    limit: Optional[int] = None,
    repair: bool = False,
    rng: Optional[random.Random] = None,
) -> PoolReport:
    """Accept the candidates a generator may actually emit into the run.

    Every candidate is checked value by value: the shape must equal the
    template's, a locus with a declared domain must hold a declared value,
    and a locus the schema does not constrain must keep the template's value
    -- the same rule ``llm_init`` applies to authored initial members, which
    is the point: one authoring line, not one per seam.

    The diversity guard rides along, because a batch is a SET and its
    degeneracies are only visible batch-wide: a candidate repeating an
    earlier member of the same batch counts as a duplicate, and one whose
    key is in *seen* (everything the run has measured) counts as archive
    overlap. Both are dropped -- they consume a pool slot and can teach the
    run nothing -- and both are counted, so a generator that has collapsed
    onto one configuration reads as ``duplicates == emitted - 1`` instead of
    as an unremarkable flat curve.

    *limit* caps how many members are considered at all, so a generator that
    answers "give me 2,000" with 200,000 costs the harness the 2,000 it
    asked for. ``PoolReport.emitted`` counts what was considered, which is
    the denominator every rate here is against.

    Every reject is also ADDRESSED, into :class:`RejectionCensus`: which
    locus, which reason, and one concrete offending sample. A counter that
    says "6,021 shape" cannot instruct a revision; "job_13 is missing from
    every candidate you emitted, sample {...}" can.

    *repair* turns the per-candidate fallback into a per-LOCUS one. Off (the
    default, and what every sealed row was measured under) a candidate with
    one bad locus is dropped whole and its pool slot is filled by a
    schema-uniform draw, so thirteen good choices are discarded with the
    fourteenth. On, the harness assembles the candidate out of whatever the
    member did address -- flat locus keys, the template's own nesting, or a
    bare sequence aligned with the loci -- and decides only the loci the
    member got wrong or left out. Repairs are accepted, but counted apart in
    ``repaired``/``repaired_loci`` and censused per locus, because a
    candidate the harness finished is not a candidate the model wrote.
    """

    if not isinstance(emitted, list):
        return PoolReport()
    template_loci = loci_of(template)
    template_fields = set(template)
    known = set(seen) if seen is not None else set()
    batch: set[str] = set()
    draw = rng if rng is not None else random.Random(0)

    accepted: List[Config] = []
    census = RejectionCensus()
    counts = {"shape": 0, "domain": 0, "duplicate": 0, "overlap": 0,
              "repaired": 0, "repaired_loci": 0}
    considered = 0
    for member in emitted:
        if limit is not None and considered >= limit:
            break
        considered += 1
        candidate, reason, locus, value = _read_candidate(
            member, template=template, template_loci=template_loci,
            template_fields=template_fields, domains=domains)
        if candidate is None and repair:
            fixed, repairs = coerce_candidate(
                member, template=template, domains=domains, rng=draw,
                loci=template_loci)
            if fixed is not None:
                for kind, loci_list in repairs.items():
                    for name in loci_list:
                        census.repaired(name)
                        if kind == "out_of_domain":
                            census.out_of_domain(name, _picked(member, name))
                counts["repaired"] += 1
                counts["repaired_loci"] += sum(len(v) for v in repairs.values())
                candidate, reason = fixed, ""
        if candidate is None:
            if reason == "domain":
                counts["domain"] += 1
                census.out_of_domain(str(locus), value)
            else:
                counts["shape"] += 1
                census.shape(reason, member)
            continue
        key = candidate_key(candidate)
        if key in batch:
            counts["duplicate"] += 1
            continue
        # Marked as seen in this batch whatever happens next, so a candidate
        # that is BOTH already measured and repeated eight times reads as one
        # overlap and seven duplicates. Two different defects, two counters.
        batch.add(key)
        if key in known:
            counts["overlap"] += 1
            continue
        accepted.append(dict(candidate))

    return PoolReport(
        accepted=tuple(accepted),
        emitted=considered,
        rejected_shape=counts["shape"],
        rejected_out_of_domain=counts["domain"],
        duplicates=counts["duplicate"],
        archive_overlap=counts["overlap"],
        repaired=counts["repaired"],
        repaired_loci=counts["repaired_loci"],
        census=census,
    )


def _read_candidate(member, *, template, template_loci, template_fields,
                    domains):
    """``(config, reason, locus, value)`` -- the value-by-value gate itself.

    ``config`` is the member unchanged when it passes. Otherwise ``reason``
    names WHICH gate it failed, in the vocabulary a revision can act on:
    ``not a mapping``, ``missing loci``, ``unexpected fields``, ``wrong
    length``, or ``domain`` with the offending locus and value.
    """

    if not isinstance(member, dict):
        return None, "not a mapping", None, None
    fields_seen = set(member)
    if fields_seen != template_fields:
        missing = sorted(template_fields - fields_seen)
        extra = sorted(fields_seen - template_fields)
        if missing and extra:
            reason = (f"missing fields {missing[:4]} and unexpected fields "
                      f"{extra[:4]}")
        elif missing:
            reason = f"missing fields {missing[:6]}"
        else:
            reason = f"unexpected fields {extra[:6]}"
        return None, reason, None, None
    try:
        member_loci = loci_of(member)
    except Exception:
        return None, "not a mapping", None, None
    if member_loci != template_loci:
        want, got = len(template_loci), len(member_loci)
        return None, (f"wrong sequence length ({got} loci, the archive's "
                      f"members have {want})"), None, None
    for locus in member_loci:
        value = read_locus(member, locus)
        domain = domains.get(str(locus)) or ()
        if domain:
            if value not in domain:
                return None, "domain", locus, value
        elif value != read_locus(template, locus):
            return None, "domain", locus, value
    return member, "", None, None


def _picked(member: Any, locus: str) -> Any:
    """The value *member* carried at *locus*, for the census's sample."""

    try:
        if isinstance(member, dict):
            if locus in member:
                return member[locus]
            if locus.endswith("]") and "[" in locus:
                field, index = locus[:-1].split("[", 1)
                return member[field][int(index)]
    except Exception:
        return None
    return None


GENERATOR_PROMPT = """You are writing the CANDIDATE GENERATOR for a black-box \
multi-objective optimizer. It calls your function every generation to draw the \
pool of configurations it will consider; you are writing the DISTRIBUTION those \
draws come from, not any particular draw.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE:
{schema}
{loci}
Write ONE Python function with EXACTLY this signature:

    {contract}

Rules:
- `domains` maps every locus to its allowed values under the current sampling
  prior; sequence fields appear per position as `name[i]`. Each configuration
  you return must have the SAME SHAPE as the archive members and take every
  value from `domains` at that locus -- anything else is validated out and
  its slot falls back to a uniform random draw.
{scaffold}- `archive` holds configurations already measured in this run (it may be
  short early on). Use it for context; do NOT return copies of it, and do not
  return the same configuration twice. A batch that collapses is measured and
  reported back to you.
- Return EXACTLY `n` configurations. `n` can be large (thousands): this is
  mass generation, so keep it cheap and vectorless -- plain loops over
  `domains`.
- Use what the parameter NAMES AND MEANINGS say about this domain to bias
  where mass lands: known good regions, couplings that must co-move,
  trade-offs worth spreading along. That knowledge is the only reason your
  sampler can beat drawing uniformly from the same domains, which is exactly
  what it is measured against.
- Derive all randomness from `seed` (e.g. `random.Random(seed)`), so a pool
  is reproducible.
- Standard library only; imports limited to: {imports}.
- No I/O, no globals, deterministic.
{limits}
Reply with ONLY one fenced Python code block and no other text."""


#: The resource contract, echoed for the same reason the domains are: a
#: function that exceeds its sandbox budget returns NOTHING, so the whole pool
#: falls back to schema-uniform draws and the mechanism contributes zero. Wave
#: D's `upms_j14_m3` telemetry is the evidence -- 7,104 candidates emitted
#: against 39,993 pool slots filled uniformly means most BATCHES emitted
#: nothing at all, which is what a timeout looks like when the counter is
#: per-candidate. An unstated budget is a budget the author cannot honour.
LIMITS_RULES = """\
- HARD RESOURCE LIMITS, enforced by the sandbox: {wall} s wall-clock, {cpu} s
  CPU and {memory} MB of memory for ONE call, at up to n={max_n}. Exceeding
  any of them returns NOTHING -- not a partial pool, nothing -- and the run
  falls back to drawing every candidate uniformly, which is exactly the
  baseline you are being measured against. Budget for the WORST case, not the
  typical one: prefer O(n) construction from `domains` to any search, sort or
  simulation over candidates, and if you want a local improvement step, cap
  its total work by a constant you choose rather than by convergence."""


GENERATOR_REVISION_PROMPT = """You previously wrote this candidate generator \
for a black-box multi-objective optimizer:

```python
{source}
```

The harness ran it, validated everything it emitted, and measured what
survived. Here is what actually happened:

{feedback}
{loci}
Revise the function. Read the numbers literally: a rejected or repaired
candidate names the LOCUS it failed at and the value it tried, so fix that
locus rather than the sampler in general; duplicates mean the sampler is
collapsing; archive overlap means it keeps re-proposing configurations
already measured; no survivors means the region it concentrates on is not
competitive and the mass should move. Same rules as before: exactly this
signature

    {contract}

exactly `n` configurations, every value from `domains` at that locus, the same
shape as the archive members, randomness derived from `seed`, standard library
only ({imports}), deterministic, no I/O.
{scaffold}{limits}
Reply with ONLY one fenced Python code block and no other text."""


GENERATOR_EVIDENCE_PROMPT = """You wrote this candidate generator for a \
black-box multi-objective optimizer:

```python
{source}
```

The optimizer has been running it and MEASURING what it drew. Here is the
evidence -- the run's own measurements, and nothing else:

{evidence}

Now reason about where to sample next, and rewrite the function so its mass
lands there. Concretely: which parameters do the measurements say actually
move the costs, and in which direction? Which regions has the run already
measured and found not competitive, so that re-proposing them wastes the rest
of the budget? Where is the front, and what is the smallest change to a front
member that has not been measured yet?

Your previous version was written before any of this was measured. It is not
being corrected for a defect -- it is being asked to use information that did
not exist when it was written. If the measurements do not support a change,
say so by returning a function that differs only where they do.

Same contract as before: exactly this signature

    {contract}

exactly `n` configurations, every value taken from `domains` at that locus,
the same shape as the archive members, randomness derived from `seed`,
standard library only ({imports}), deterministic, no I/O.
{scaffold}{limits}
Reply with ONLY one fenced Python code block and no other text."""


def _locus_block(domains: Optional[Mapping[str, Sequence[Any]]]) -> str:
    """The per-locus domain echo, as a prompt section (empty when unknown)."""

    if not domains:
        return ""
    echo = render_domain_echo(domains)
    if not echo:
        return ""
    return ("\nLOCI AND THEIR ADMISSIBLE VALUES (the exact `domains` mapping "
            "you will be passed;\nthese key names ARE the shape -- a "
            "configuration has exactly these loci and no others):\n"
            f"{echo}\n")


def _scaffold_block(scaffold: bool) -> str:
    return (SCAFFOLD_RULES + "\n") if scaffold else ""


def _limits_block(limits: Any, max_n: Optional[int]) -> str:
    """The sandbox's own budget, echoed (empty when the caller knows none)."""

    if limits is None or not max_n:
        return ""
    try:
        return LIMITS_RULES.format(
            wall=f"{float(limits.wall_time_s):g}",
            cpu=f"{float(limits.cpu_seconds):g}",
            memory=int(int(limits.memory_bytes) / (1024 * 1024)),
            max_n=int(max_n)) + "\n"
    except (AttributeError, TypeError, ValueError):
        return ""


def author_generator(
    complete: Callable[[str], str],
    *,
    objectives: Sequence[ObjectiveSpec],
    schema_text: str,
    attempts: int = 2,
    telemetry: Optional[AuthorTelemetry] = None,
    domains: Optional[Mapping[str, Sequence[Any]]] = None,
    scaffold: bool = True,
    limits: Any = None,
    max_n: Optional[int] = None,
) -> Optional[AuthoredArtifact]:
    """Ask the model to write ``propose``; accept whole or not at all.

    *domains* is the per-locus admissible set the run will actually pass, so
    the prompt can ECHO it rather than leave the model to infer per-position
    domains from a field-level card -- the difference that turns an
    out-of-domain value from a guess into a prompt failure. *scaffold*
    advertises the in-sandbox emit harness (``build``), which is what makes a
    shape error impossible to construct rather than caught after the fact.
    """

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["generator"]
    prompt = GENERATOR_PROMPT.format(
        goals="\n".join(f"  {s.name}: {s.goal}imise" for s in objectives),
        schema=schema_text,
        loci=_locus_block(domains),
        scaffold=_scaffold_block(scaffold),
        limits=_limits_block(limits, max_n),
        contract=contract.description,
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
    )
    return _author(complete, prompt, contract=contract, attempts=attempts,
                   telemetry=tel, name="llm_generator")


def revise_generator(
    complete: Callable[[str], str],
    *,
    artifact: AuthoredArtifact,
    feedback: str,
    attempts: int = 1,
    telemetry: Optional[AuthorTelemetry] = None,
    domains: Optional[Mapping[str, Sequence[Any]]] = None,
    scaffold: bool = True,
    limits: Any = None,
    max_n: Optional[int] = None,
) -> Optional[AuthoredArtifact]:
    """One revision round: the artifact, its measured behaviour, a rewrite.

    The gate treats a revision exactly like a fresh authorship -- fenced
    block only, import allowlist, correct entry point, whole-reply rejection
    -- so a model that answers a revision with prose, or with code that
    imports the filesystem, keeps the generator it already had.
    """

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["generator"]
    prompt = GENERATOR_REVISION_PROMPT.format(
        source=artifact.source,
        feedback=feedback,
        loci=_locus_block(domains),
        scaffold=_scaffold_block(scaffold),
        limits=_limits_block(limits, max_n),
        contract=contract.description,
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
    )
    return _author(complete, prompt, contract=contract, attempts=attempts,
                   telemetry=tel, name=f"{artifact.name}_rev")


def reauthor_generator(
    complete: Callable[[str], str],
    *,
    artifact: AuthoredArtifact,
    evidence: str,
    attempts: int = 1,
    telemetry: Optional[AuthorTelemetry] = None,
    scaffold: bool = True,
    limits: Any = None,
    max_n: Optional[int] = None,
) -> Optional[AuthoredArtifact]:
    """Re-author the sampler against the run's MEASURED trace.

    Structurally identical to :func:`revise_generator` -- same contract, same
    whole-reply gate, same degradation to the artifact already in hand -- and
    different in the one way that matters: the prompt carries measurements
    instead of emission counters, so the model is asked to reason about the
    search rather than to repair its own output. The scaffold and resource
    contracts are echoed exactly as they are for authorship and revision: a
    re-authored sampler runs in the same sandbox as the one it replaces.
    """

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["generator"]
    prompt = GENERATOR_EVIDENCE_PROMPT.format(
        source=artifact.source,
        evidence=evidence,
        scaffold=_scaffold_block(scaffold),
        limits=_limits_block(limits, max_n),
        contract=contract.description,
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
    )
    return _author(complete, prompt, contract=contract, attempts=attempts,
                   telemetry=tel, name=f"{artifact.name}_evidence")


def _author(complete, prompt, *, contract, attempts, telemetry, name):
    for _attempt in range(max(1, attempts)):
        telemetry.calls += 1
        try:
            text = complete(prompt)
        except Exception:
            telemetry.errors += 1
            continue
        source = accept_block(text, contract=contract, telemetry=telemetry)
        if source is None:
            continue
        telemetry.accepted += 1
        telemetry.sources.append(source)
        return authored_artifact(contract.kind, source, name=name,
                                 authored_by="llm")
    return None


#: How many offending loci one feedback block names before it stops. A
#: revision cannot act on four hundred addresses; it can act on the worst few.
FEEDBACK_LOCI = 6


def _defect_lines(
    census: Optional[RejectionCensus],
    domains: Optional[Mapping[str, Sequence[Any]]] = None,
) -> List[str]:
    """WHICH loci failed and WHY, worst first, each with a real sample."""

    if census is None or census.empty:
        return []
    lines: List[str] = ["  WHERE IT FAILED (the harness's own addresses):"]
    for reason, count in sorted(census.shape_reasons.items(),
                                key=lambda kv: -kv[1])[:FEEDBACK_LOCI]:
        sample = census.samples.get(f"shape:{reason}")
        lines.append(f"    shape -- {reason}: {count} candidate(s); "
                     f"you emitted {json_compact(sample)}")
    ranked = sorted(census.out_of_domain_by_locus.items(),
                    key=lambda kv: -kv[1])
    for locus, count in ranked[:FEEDBACK_LOCI]:
        sample = census.samples.get(f"domain:{locus}")
        allowed = list((domains or {}).get(locus) or ())
        rendered = (f"; its domain is {allowed[:8]}"
                    + (f" ({len(allowed)} values)" if len(allowed) > 8 else "")
                    ) if allowed else ""
        lines.append(f"    locus {locus} -- out of domain {count} time(s); "
                     f"you used {json_compact(sample)}{rendered}")
    if len(ranked) > FEEDBACK_LOCI:
        lines.append(f"    ... and {len(ranked) - FEEDBACK_LOCI} further loci "
                     f"out of domain")
    repaired = sorted(census.repaired_by_locus.items(), key=lambda kv: -kv[1])
    if repaired:
        worst = ", ".join(f"{locus} ({count})"
                          for locus, count in repaired[:FEEDBACK_LOCI])
        lines.append(f"    the harness had to DECIDE these loci for you: "
                     f"{worst}")
    return lines


def _edit_lines(rejected_edits: Sequence[Mapping[str, Any]]) -> List[str]:
    """The rejected-edit memory: fixes already tried that did not fix it.

    Wave D measured revision firing on 77 of 80 cells on the broken
    instances and repairing none of them. A revision loop with no memory of
    its own failures can only re-propose them; naming the edit, the defect it
    was supposed to fix, and the fact that the defect survived it is the
    cheapest thing that makes the next attempt different.
    """

    if not rejected_edits:
        return []
    lines = ["  EDITS ALREADY TRIED THAT DID NOT FIX THIS -- do not repeat "
             "them or anything equivalent:"]
    for edit in rejected_edits[-3:]:
        lines.append(
            f"    revision {edit.get('revision')} (source {edit.get('sha')}): "
            f"defect rate {float(edit.get('before', 0.0)):.0%} -> "
            f"{float(edit.get('after', 0.0)):.0%}, and the same loci still "
            f"fail ({edit.get('signature')}).")
        excerpt = str(edit.get("excerpt") or "").strip()
        if excerpt:
            lines.append("      it looked like: "
                         + " ".join(excerpt.split())[:240])
    return lines


def render_generation_feedback(
    telemetry: GeneratorTelemetry,
    last: Optional[PoolReport] = None,
    survivors: Sequence[Tuple[Config, Mapping[str, float]]] = (),
    *,
    census: Optional[RejectionCensus] = None,
    domains: Optional[Mapping[str, Sequence[Any]]] = None,
    rejected_edits: Sequence[Mapping[str, Any]] = (),
) -> str:
    """The measured story a generator revision needs, as text.

    Three things, all counted: how much of what it emitted the harness could
    use, how much of it was novel, and what happened to the candidates that
    were measured. Survivors are shown rather than "the best" -- ranking
    candidates across objectives would need weights nobody declared, while
    surviving truncation is the run's own weight-free verdict.

    Then the two things Wave D's counters could not say, and without which
    revision repaired nothing: WHICH locus rejected and WHY, with a value the
    model will recognise as its own (*census*, *domains*), and which edits
    have already been tried against this same defect and failed
    (*rejected_edits*).
    """

    lines = [
        f"  batches generated: {telemetry.batches}"
        f" (runtime failures: {telemetry.runtime_failures})",
        f"  candidates emitted: {telemetry.emitted}; "
        f"accepted by the harness: {telemetry.accepted}",
        f"  rejected -- wrong shape: {telemetry.rejected_shape}; "
        f"value outside its declared domain: "
        f"{telemetry.rejected_out_of_domain}",
        f"  dropped -- duplicate within the batch: {telemetry.duplicates}; "
        f"already measured in this run: {telemetry.archive_overlap}",
        f"  pool slots the harness had to fill with uniform random draws: "
        f"{telemetry.filled_uniform}",
        f"  of yours that were measured: {telemetry.measured}; "
        f"survived selection into the next population: {telemetry.survived}",
    ]
    if telemetry.repaired or telemetry.repaired_loci:
        lines.append(
            f"  candidates the harness had to ASSEMBLE for you rather than "
            f"reject: {telemetry.repaired} "
            f"({telemetry.repaired_loci} individual loci decided for you)")
    if (telemetry.scaffold_filled or telemetry.scaffold_out_of_domain
            or telemetry.scaffold_unknown_locus):
        lines.append(
            f"  inside your own code, `build` filled "
            f"{telemetry.scaffold_filled} locus/loci you left unset, "
            f"overrode {telemetry.scaffold_out_of_domain} out-of-domain "
            f"value(s) and ignored {telemetry.scaffold_unknown_locus} locus "
            f"name(s) the schema does not declare")
    if last is not None and last.emitted:
        lines.append(
            f"  most recent batch: {last.duplicate_rate:.0%} duplicates, "
            f"{last.archive_overlap_rate:.0%} already measured, "
            f"{last.novelty_rate:.0%} usable and new")
    lines.extend(_defect_lines(
        census if census is not None
        else (last.census if last is not None else None), domains))
    lines.extend(_edit_lines(rejected_edits))
    for config, objectives in survivors:
        rendered = ", ".join(f"{k}={float(v):.6g}"
                             for k, v in sorted(objectives.items()))
        lines.append(f"  survived: {json_compact(config)} -> {rendered}")
    return "\n".join(lines)


class AuthoredGenerator:
    """The authored sampler as a loop policy: mass generation, then the guard.

    One call per generation produces the whole pool out of process; the
    harness validates it, drops what collapsed, fills any shortfall
    schema-uniformly, and hands back exactly ``pool_for(want)``
    configurations. Nothing here can reach an evaluation: the pool is
    candidates, and the loop measures at most the ``want`` it could already
    afford.
    """

    def __init__(
        self,
        artifact: AuthoredArtifact,
        runtime: Any,
        *,
        pool_factor: int = 4,
        pool_size: int = 0,
        max_pool: int = MAX_POOL,
        archive_shown: int = ARCHIVE_SHOWN,
        revise: Optional[Callable[[AuthoredArtifact, str],
                                  Optional[AuthoredArtifact]]] = None,
        max_revisions: int = 1,
        min_measured_for_revision: int = 4,
        min_novelty: float = 0.5,
        scaffold: bool = True,
        repair: bool = True,
        revision_guard: bool = False,
        shrink_on_overrun: int = 4,
        objectives: Sequence[ObjectiveSpec] = (),
        reauthor: Optional[Callable[[AuthoredArtifact, str],
                                    Optional[AuthoredArtifact]]] = None,
        reauthor_every: int = 0,
        max_reauthorings: int = 0,
        evidence_view: Optional[Callable[[Sequence[MeasuredRow]],
                                         Sequence[MeasuredRow]]] = None,
        evidence_front_shown: int = 8,
        evidence_effects_shown: int = 8,
        prior_author: Optional[Callable[[str], str]] = None,
        max_priors: int = 1,
        prior_max_narrowing_log10: float = 2.0,
        prior_unwind_batches: int = 2,
    ) -> None:
        if pool_factor < 1:
            raise ValueError(f"pool_factor must be at least 1, got {pool_factor}")
        if pool_size < 0:
            raise ValueError(f"pool_size must be non-negative, got {pool_size}")
        if reauthor_every < 0:
            raise ValueError(
                f"reauthor_every must be non-negative, got {reauthor_every}")
        if prior_author is not None and reauthor_every <= 0:
            # The evidence channel has one cadence and both consumers ride it.
            # A prior asked for on no cadence would fire never or every
            # generation depending on who read the code, which is exactly the
            # magic number this knob exists to replace.
            raise ValueError(
                "a locus prior is authored from the measured trace on the "
                "reauthor_every cadence; set reauthor_every > 0")
        self.artifact = artifact
        self.runtime = runtime
        self.pool_factor = int(pool_factor)
        self.pool_size = int(pool_size)
        self.max_pool = int(max_pool)
        self.archive_shown = int(archive_shown)
        self.revise = revise
        self.max_revisions = int(max_revisions)
        self.min_measured_for_revision = int(min_measured_for_revision)
        self.min_novelty = float(min_novelty)
        #: Ship the emit harness into the sandbox, so the authored code
        #: constructs candidates locus by locus instead of transcribing a
        #: shape. Shape was 6,021 of 7,104 emissions on `upms_j14_m3`.
        self.scaffold = bool(scaffold)
        #: Assemble a candidate out of whatever the emission got right rather
        #: than dropping it whole -- a per-LOCUS fallback in place of a
        #: per-CANDIDATE one. Off restores the sealed-row semantics exactly.
        self.repair = bool(repair)
        #: Keep a revision only if a frozen replay says it MEASURABLY helped
        #: (see :meth:`_guard_admits`). Off by default: the one-shot and
        #: capped-revision arms are what every sealed row is defined on.
        self.revision_guard = bool(revision_guard)
        #: Divisor for the one retry a resource overrun gets. 0 or 1 disables
        #: it and a timeout costs the whole pool, as it did when Wave D
        #: measured 39,993 uniformly-filled slots against 7,104 emissions.
        self.shrink_on_overrun = int(shrink_on_overrun)
        #: The MEASUREMENT-CONDITIONED channel. ``reauthor_every`` is a
        #: cadence in CHARGED EVALUATIONS -- a declared, typed knob rather
        #: than a magic number -- and ``0`` (the default) is the seam every
        #: sealed row to date ran: no evidence call ever fires.
        self.objectives = tuple(objectives)
        self.reauthor = reauthor
        self.reauthor_every = int(reauthor_every)
        self.max_reauthorings = int(max_reauthorings)
        #: What the model is SHOWN. The identity view is the product; a view
        #: returning another run's rows is the shuffled-evidence control, and
        #: it is a parameter rather than a patch precisely so the control can
        #: be built without editing this file.
        self.evidence_view = evidence_view
        self.evidence_front_shown = int(evidence_front_shown)
        self.evidence_effects_shown = int(evidence_effects_shown)
        #: The locus-importance channel, and the gate that refuses it.
        self.prior_author = prior_author
        self.max_priors = int(max_priors)
        self.prior_max_narrowing_log10 = float(prior_max_narrowing_log10)
        self.prior_unwind_batches = int(prior_unwind_batches)
        self.telemetry = GeneratorTelemetry()
        self.mechanism = "authored_generator"
        self.authored_by = artifact.authored_by
        self.last_report: Optional[PoolReport] = None
        self.census = RejectionCensus()
        self._seen: set[str] = set()
        self._survivors: List[Tuple[Config, Mapping[str, float]]] = []
        self._domains: Dict[str, List[Any]] = {}
        self._last_call: Optional[Tuple[List[Config], int, Dict[str, List[Any]],
                                        int, Config]] = None
        self._rejected_edits: List[Dict[str, Any]] = []
        self._pending_edit: Optional[Dict[str, Any]] = None
        #: The measured trace, in measurement order. This is the evidence, and
        #: it is the ONLY thing this class knows about outcomes: it arrives
        #: through `record_measured`, which the loop already calls, so the
        #: generator still never sees the problem, the evaluator or the cache.
        self._rows: List[MeasuredRow] = []
        self._evidence_at = 0
        self._prior_asked_at = -1
        self._prior: Any = None
        self._prior_batches = 0
        self._survived_at_prior = 0
        #: One record per evidence-conditioned call: what was shown (digest
        #: and row count), what came back, and whether it was accepted.
        #: Telemetry as correctness -- a run cannot claim this channel fired
        #: without the record that says what it saw.
        self.evidence_log: List[Dict[str, Any]] = []
        self._noted = 0

    # -- sizing -------------------------------------------------------------

    def pool_for(self, want: int) -> int:
        """How many candidates one generation asks for. Never below *want*."""

        size = self.pool_size or self.pool_factor * max(1, want)
        return max(1, min(self.max_pool, max(int(want), size)))

    # -- the archive it is conditioned on and measured against --------------

    def note_archive(self, configs: Sequence[Config]) -> None:
        """Record configurations the run has measured. No quality claim."""

        for config in configs:
            self._seen.add(candidate_key(config))

    def record_measured(
        self,
        config: Config,
        *,
        survived: bool,
        objectives: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Credit at survival time, exactly as the operator portfolio does."""

        self._seen.add(candidate_key(config))
        self.telemetry.measured += 1
        if objectives is not None:
            # The measured trace: kept whether or not it survived, because
            # "this region was measured and is NOT competitive" is exactly the
            # evidence a survivor list cannot carry.
            self._rows.append((dict(config), dict(objectives), bool(survived)))
        if survived:
            self.telemetry.survived += 1
            if objectives is not None:
                self._survivors.append((dict(config), dict(objectives)))
                del self._survivors[:-3]

    # -- generation ---------------------------------------------------------

    def propose(
        self,
        *,
        template: Config,
        candidate_model: Any,
        restriction: Any,
        archive: Sequence[Config],
        want: int,
        rng: random.Random,
        seed: int = 0,
    ) -> List[Config]:
        """A validated pool of ``pool_for(want)`` configurations.

        The head of the pool is what the loop would measure with no screen at
        all, so a screen that follows keeps its exploration floor over the
        generator's OWN first picks rather than over an unrelated draw.
        """

        n = self.pool_for(want)
        declared = {
            str(locus): list(locus_domain(candidate_model, locus,
                                          restriction=restriction))
            for locus in loci_of(template)
        }
        self._domains = declared
        # Search-progress triggers, on the DECLARED domains: the evidence and
        # the prior both describe the space the problem published, not a space
        # a previous prior already narrowed, or a second prior would compound
        # the first one's bet without ever measuring it.
        # ONE cadence tick, read once and consumed by both channels: asking
        # each of them separately would let whichever ran first advance the
        # anchor and starve the other, which is a rule nobody declared.
        due = self._due()
        self._maybe_reauthor(declared, due)
        self._maybe_author_prior(declared, due)
        if due:
            self._evidence_at = self.telemetry.measured
        domains = self._effective_domains(declared)
        shown = [dict(config) for config in list(archive)[:self.archive_shown]]
        self._maybe_revise()
        self._last_call = (shown, n, domains, int(seed), dict(template))

        self.telemetry.batches += 1
        # The generator SAMPLES from the (possibly biased) domains and is
        # VALIDATED against the declared ones. A prior is guidance about where
        # to spend, not a new definition of what is legal, so a candidate
        # outside the prior but inside the schema is admitted rather than
        # counted as a defect -- otherwise installing a prior would
        # manufacture rejections and fire the defect-repair channel on a
        # generator that did exactly what it was asked.
        report = self._run(self.artifact, shown, n, domains, seed,
                           template=template, rng=rng, count=True,
                           validate_domains=declared)
        self.last_report = report
        self.census.merge(report.census)
        self._score_pending_edit(report)
        self.telemetry.emitted += report.emitted
        self.telemetry.accepted += len(report.accepted)
        self.telemetry.rejected_shape += report.rejected_shape
        self.telemetry.rejected_out_of_domain += report.rejected_out_of_domain
        self.telemetry.duplicates += report.duplicates
        self.telemetry.archive_overlap += report.archive_overlap
        self.telemetry.repaired += report.repaired
        self.telemetry.repaired_loci += report.repaired_loci

        pool = [dict(config) for config in report.accepted[:n]]
        while len(pool) < n:
            pool.append(uniform_candidate(template, candidate_model, rng=rng,
                                          restriction=restriction))
            self.telemetry.filled_uniform += 1
        return pool

    def _run(self, artifact, archive, n, domains, seed, *, template, rng,
             count: bool,
             validate_domains: Optional[Mapping[str, Sequence[Any]]] = None,
             ) -> PoolReport:
        """One emission through the scaffold, validated. Optionally counted.

        *count* is false for the revision guard's frozen replay, which must
        measure a challenger without the run's telemetry recording an
        emission the loop never saw.

        *validate_domains*, when given, is what the pool is judged against --
        the DECLARED domains -- while *domains* is what the sampler draws
        from (a weighted prior may have biased them). Sampling guidance must
        never redefine what is legal.
        """

        rows = self._emit(artifact, archive, n, domains, seed,
                          template=template, count=count)
        return validate_pool(
            rows, template=template,
            domains=validate_domains if validate_domains is not None else domains,
            seen=self._seen, limit=n, repair=self.repair,
            rng=rng if rng is not None else random.Random(seed))

    def _emit(self, artifact, archive, n, domains, seed, *, template,
              count: bool = True) -> Any:
        prelude = (scaffold_prelude(template, domains, nonce=int(seed))
                   if self.scaffold else None)
        try:
            outcome = self.runtime.call(
                artifact, [[archive, int(n), domains, int(seed)]],
                prelude=prelude, notes_global=NOTES_GLOBAL)
        except TypeError:
            # A runtime that predates the prelude channel: the artifact still
            # runs, the scaffold simply is not there, and the harness-side
            # repair remains the only guard. Degrade, never fail.
            try:
                outcome = self.runtime.call(
                    artifact, [[archive, int(n), domains, int(seed)]])
            except Exception:
                if count:
                    self.telemetry.runtime_failures += 1
                return []
        except Exception:                    # a runtime that cannot even ship
            if count:                                # the call is a countable
                self.telemetry.runtime_failures += 1  # event, not an emergency
            return []
        if count:
            self._absorb_notes(getattr(outcome, "notes", None))
        if (not outcome.ok and self.shrink_on_overrun > 1
                and outcome.status in ("timeout", "memory")
                and int(n) > self.shrink_on_overrun):
            # A resource overrun is the one failure whose CAUSE the harness
            # can act on: `propose` is a distribution, so asking it for fewer
            # draws is the same request at a fraction of the work. A quarter
            # of a guided pool beats none of one, the shortfall still falls
            # back to schema-uniform, and both events stay counted.
            if count:
                self.telemetry.runtime_failures += 1
                self.telemetry.runtime_retries += 1
            smaller = max(1, int(n) // self.shrink_on_overrun)
            try:
                outcome = self.runtime.call(
                    artifact, [[archive, smaller, domains, int(seed)]],
                    prelude=prelude, notes_global=NOTES_GLOBAL)
            except Exception:
                return []
            if count and outcome.ok:
                self.telemetry.runtime_recovered += 1
                self._absorb_notes(getattr(outcome, "notes", None))
            if not outcome.ok:
                return []
            [rows] = outcome.results
            return rows if isinstance(rows, list) else []
        if not outcome.ok:
            if count:
                self.telemetry.runtime_failures += 1
            return []
        [rows] = outcome.results
        if not isinstance(rows, list):
            if count:
                self.telemetry.runtime_failures += 1
            return []
        return rows

    def _absorb_notes(self, notes: Any) -> None:
        """The scaffold's own counters, from inside the sandbox."""

        if not isinstance(notes, Mapping):
            return
        self.telemetry.scaffold_filled += int(notes.get("filled") or 0)
        self.telemetry.scaffold_out_of_domain += int(
            notes.get("out_of_domain") or 0)
        self.telemetry.scaffold_unknown_locus += int(
            notes.get("unknown_locus") or 0)
        by_locus = notes.get("by_locus")
        if isinstance(by_locus, Mapping):
            for locus, row in by_locus.items():
                if not isinstance(row, Mapping):
                    continue
                name = str(locus)
                for _ in range(int(row.get("filled") or 0)):
                    self.census.repaired(name)
                count = int(row.get("out_of_domain") or 0)
                if count:
                    self.census.out_of_domain_by_locus[name] = (
                        self.census.out_of_domain_by_locus.get(name, 0) + count)
        for sample in (notes.get("samples") or ()):
            if isinstance(sample, Mapping) and "locus" in sample:
                self.census.sample(str(sample["locus"]), sample.get("value"))

    # -- revision from measured feedback ------------------------------------

    def deficient(self) -> bool:
        """The preregistered trigger: has the harness MEASURED a defect?

        Three kinds, in order of how little interpretation they need. A
        rejected candidate or a runtime failure is a broken contract, however
        rare. A batch whose novelty falls under ``min_novelty`` has collapsed
        -- onto itself or onto the archive -- which is different from the
        occasional collision any honest sampler makes in a small space, and
        the threshold is what keeps those two apart. And a generator whose
        measured children never survive is futile even when it is faultless.

        A revision fires on evidence or not at all: time passing is not
        evidence, and W3 measured that revision LEVELS the rungs, so it must
        never fire quietly on a generator that is working.
        """

        tel = self.telemetry
        if tel.batches == 0:
            return False
        if (tel.rejected_shape or tel.rejected_out_of_domain
                or tel.runtime_failures or tel.repaired
                or tel.scaffold_out_of_domain or tel.scaffold_unknown_locus):
            return True
        last = self.last_report
        if (last is not None and last.emitted
                and last.novelty_rate < self.min_novelty):
            return True
        return (tel.measured >= self.min_measured_for_revision
                and tel.survived == 0)

    def _maybe_revise(self) -> None:
        if self.revise is None or self.telemetry.revisions >= self.max_revisions:
            return
        if not self.deficient():
            return
        self.telemetry.revisions += 1
        feedback = self.feedback()
        try:
            replacement = self.revise(self.artifact, feedback)
        except Exception:                    # a revision must not kill a run
            replacement = None
        if replacement is None:
            return
        if self.revision_guard and not self._guard_admits(replacement):
            self.telemetry.revisions_rejected += 1
            self._remember_rejected_edit(replacement, guarded=True)
            return
        self._pending_edit = {
            "revision": self.telemetry.revisions,
            "sha": replacement.source_sha256[:8],
            "excerpt": replacement.source[:400],
            "before": (self.last_report.defect_rate
                       if self.last_report is not None else 0.0),
            "signature": self.census.signature(),
        }
        self.telemetry.revisions_accepted += 1
        self.artifact = replacement

    def feedback(self) -> str:
        """The measured story this generator would hand a revision."""

        return render_generation_feedback(
            self.telemetry, self.last_report, self._survivors,
            census=self.census, domains=self._domains,
            rejected_edits=self._rejected_edits)

    # -- the guard, and the memory of what it (or measurement) rejected -----

    def _guard_admits(self, replacement: AuthoredArtifact) -> bool:
        """Does a FROZEN replay say the revision measurably helped?

        The generator seam never sees the problem or the evaluation cache, so
        the only honest validation available to it is its own emission,
        replayed against the identical inputs the incumbent was last measured
        on: same archive, same ``n``, same domains, same seed. Admission takes
        a conjunction, so a revision cannot buy defect reduction with
        collapse: the defect rate must strictly fall AND the novelty rate --
        the frozen-validation score, the fraction of the batch that was
        usable and new -- must not fall.

        No model call and no evaluation is spent here; the incumbent's side of
        the comparison is the batch already measured.
        """

        incumbent = self.last_report
        if self._last_call is None or incumbent is None:
            return True                      # nothing to compare against yet
        archive, n, domains, seed, template = self._last_call
        try:
            trial = self._run(replacement, archive, n, domains, seed,
                              template=template, rng=random.Random(seed),
                              count=False)
        except Exception:
            return False
        if not trial.emitted:
            return False
        return (trial.defect_rate < incumbent.defect_rate
                and trial.novelty_rate >= incumbent.novelty_rate)

    def _remember_rejected_edit(self, artifact: AuthoredArtifact, *,
                                guarded: bool, after: float = -1.0) -> None:
        before = (self.last_report.defect_rate
                  if self.last_report is not None else 0.0)
        self._rejected_edits.append({
            "revision": self.telemetry.revisions,
            "sha": artifact.source_sha256[:8],
            "excerpt": artifact.source[:400],
            "before": before,
            "after": before if after < 0 else after,
            "signature": self.census.signature(),
            "guarded": bool(guarded),
        })
        del self._rejected_edits[:-3]

    def _score_pending_edit(self, report: PoolReport) -> None:
        """Did the revision we accepted last time actually fix anything?

        Measured on the first batch the replacement emitted. If the defect
        rate did not fall, the edit joins the rejected-edit memory and every
        later revision is told, by name, that it was tried and failed.
        """

        pending, self._pending_edit = self._pending_edit, None
        if pending is None:
            return
        if report.defect_rate < float(pending["before"]):
            return
        self.telemetry.revisions_rejected += 1
        pending["after"] = report.defect_rate
        pending["guarded"] = False
        self._rejected_edits.append(pending)
        del self._rejected_edits[:-3]

    # -- the SEARCH-PROGRESS channel: reasoning over measurements ------------

    def _evidence_rows(self) -> List[MeasuredRow]:
        """The rows the model will be shown. Identity, unless a view is set."""

        rows: Sequence[MeasuredRow] = tuple(self._rows)
        if self.evidence_view is not None:
            try:
                rows = self.evidence_view(rows)
            except Exception:               # a control that throws must not
                rows = ()                   # be able to kill a measurement
        return [row for row in rows]

    def _render_evidence(self, rows, domains) -> str:
        return render_measurement_evidence(
            rows, self.objectives, domains,
            front_shown=self.evidence_front_shown,
            effects_shown=self.evidence_effects_shown,
            charged=self.telemetry.measured)

    def _due(self) -> bool:
        """Has the archive grown by ``reauthor_every`` charged evaluations?"""

        return (self.reauthor_every > 0
                and self.telemetry.measured - self._evidence_at
                >= self.reauthor_every)

    def _log(self, kind: str, *, rows: int, evidence: str,
             emitted: Optional[str], accepted: bool, **extra: Any) -> None:
        record: Dict[str, Any] = {
            "kind": kind,
            "at_measured": int(self.telemetry.measured),
            "rows_shown": int(rows),
            "evidence_sha256": evidence_digest(evidence),
            "evidence_chars": len(evidence),
            "emitted": emitted,
            "accepted": bool(accepted),
        }
        record.update(extra)
        self.evidence_log.append(record)

    def _maybe_reauthor(self, domains: Mapping[str, Sequence[Any]],
                        due: bool) -> None:
        """Re-author the sampler against the measured trace, on cadence.

        The trigger is SEARCH PROGRESS, not an emission defect: a generator
        that emits perfectly valid candidates out of a region the run has
        already measured to be uncompetitive is never deficient, and is
        exactly the case the defect trigger cannot see.
        """

        if (self.reauthor is None or not due
                or self.telemetry.reauthorings >= self.max_reauthorings):
            return
        rows = self._evidence_rows()
        if not rows:
            return
        self.telemetry.reauthorings += 1
        self.telemetry.evidence_rows_shown += len(rows)
        evidence = self._render_evidence(rows, domains)
        try:
            replacement = self.reauthor(self.artifact, evidence)
        except Exception:               # a re-authoring must not kill a run
            replacement = None
        self._log("reauthor", rows=len(rows), evidence=evidence,
                  emitted=(None if replacement is None
                           else replacement.source_sha256),
                  accepted=replacement is not None,
                  replaced=self.artifact.source_sha256)
        if replacement is None:
            return
        self.telemetry.reauthorings_accepted += 1
        self.artifact = replacement

    def _maybe_author_prior(self, domains: Mapping[str, Sequence[Any]],
                            due: bool) -> None:
        """Ask which loci matter, type the answer, and let the GATE refuse it."""

        self._unwind_prior_if_it_stopped_paying()
        if (self.prior_author is None or not due
                or self._prior is not None
                or self.telemetry.priors_proposed >= self.max_priors):
            return
        rows = self._evidence_rows()
        if not rows:
            return
        self.telemetry.priors_proposed += 1
        evidence = self._render_evidence(rows, domains)
        prompt = LOCUS_PRIOR_PROMPT.format(
            goals="\n".join(f"  {s.name}: {s.goal}imise" for s in self.objectives),
            domains="\n".join(
                f"  {name}: {json_compact(list(values))}"
                for name, values in sorted(dict(domains).items())),
            evidence=evidence)
        try:
            reply = self.prior_author(prompt)
        except Exception:
            reply = ""
        verdict = admit_locus_prior(
            parse_locus_prior(reply),
            domains=domains,
            front=front_of(rows, self.objectives),
            max_narrowing_log10=self.prior_max_narrowing_log10)
        self._log("locus_prior", rows=len(rows), evidence=evidence,
                  emitted=(None if verdict.prior is None
                           else evidence_digest(json_compact(
                               verdict.prior.as_note()))),
                  accepted=verdict.admitted, verdict=verdict.as_note())
        if not verdict.admitted:
            self.telemetry.priors_refused += 1
            return
        self.telemetry.priors_admitted += 1
        self._prior = verdict.prior
        self._prior_batches = 0
        self._survived_at_prior = self.telemetry.survived

    def _effective_domains(
        self, declared: Mapping[str, Sequence[Any]]
    ) -> Dict[str, List[Any]]:
        if self._prior is None:
            return {k: list(v) for k, v in dict(declared).items()}
        self._prior_batches += 1
        return apply_locus_prior(declared, self._prior)

    def _unwind_prior_if_it_stopped_paying(self) -> None:
        """An admitted prior is still a bet, and a bet must be checkable.

        The gate refuses a prior that excludes a measured front member, which
        is the only unrecoverable failure. Everything else it can only be
        wrong about, so the prior is held only while it pays: after
        ``prior_unwind_batches`` generations drawn under it with not one new
        survivor, it is dropped and the run finishes on the declared domains.
        """

        if self._prior is None or self.prior_unwind_batches <= 0:
            return
        if self._prior_batches < self.prior_unwind_batches:
            return
        if self.telemetry.survived > self._survived_at_prior:
            return
        self._prior = None
        self.telemetry.priors_unwound += 1

    def note(self) -> Dict[str, Any]:
        """The per-generation history record for the last batch."""

        note = (self.last_report.as_note() if self.last_report is not None
                else PoolReport().as_note())
        note["artifact"] = f"{self.artifact.name}:{self.artifact.source_sha256[:8]}"
        note["revisions"] = self.telemetry.revisions_accepted
        if self.reauthor_every > 0:
            # What the model saw and what it emitted, in the run's own record.
            # Only what is NEW since the last generation's note: the full log
            # stays on the object, and the history is a diary rather than n
            # copies of the same list.
            fresh = self.evidence_log[self._noted:]
            self._noted = len(self.evidence_log)
            note["evidence"] = [dict(record) for record in fresh]
            note["prior_active"] = self._prior is not None
        return note

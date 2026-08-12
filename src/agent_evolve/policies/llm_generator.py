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
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.authored import CONTRACTS, AuthoredArtifact, authored_artifact
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.authored_worker import ALLOWED_IMPORTS
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

__all__ = [
    "GeneratorTelemetry",
    "PoolReport",
    "AuthoredGenerator",
    "author_generator",
    "revise_generator",
    "render_generation_feedback",
    "validate_pool",
    "candidate_key",
    "GENERATOR_PROMPT",
    "GENERATOR_REVISION_PROMPT",
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

    def as_dict(self) -> dict[str, int]:
        return {f.name: int(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class PoolReport:
    """One mass-generation batch, as the harness received it."""

    accepted: Tuple[Config, ...] = ()
    emitted: int = 0
    rejected_shape: int = 0
    rejected_out_of_domain: int = 0
    duplicates: int = 0
    archive_overlap: int = 0

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

    def as_note(self) -> Dict[str, Any]:
        """The per-generation history record: counts plus the guard's rates."""

        return {
            "emitted": self.emitted,
            "accepted": len(self.accepted),
            "rejected_shape": self.rejected_shape,
            "rejected_out_of_domain": self.rejected_out_of_domain,
            "duplicates": self.duplicates,
            "archive_overlap": self.archive_overlap,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "duplicate_rate": round(self.duplicate_rate, 4),
            "archive_overlap_rate": round(self.archive_overlap_rate, 4),
            "novelty_rate": round(self.novelty_rate, 4),
        }


def validate_pool(
    emitted: Any,
    *,
    template: Config,
    domains: Mapping[str, Sequence[Any]],
    seen: Optional[Any] = None,
    limit: Optional[int] = None,
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
    """

    if not isinstance(emitted, list):
        return PoolReport()
    template_loci = loci_of(template)
    template_fields = set(template)
    known = set(seen) if seen is not None else set()
    batch: set[str] = set()

    accepted: List[Config] = []
    counts = {"shape": 0, "domain": 0, "duplicate": 0, "overlap": 0}
    considered = 0
    for member in emitted:
        if limit is not None and considered >= limit:
            break
        considered += 1
        if not isinstance(member, dict) or set(member) != template_fields:
            counts["shape"] += 1
            continue
        try:
            member_loci = loci_of(member)
        except Exception:
            counts["shape"] += 1
            continue
        if member_loci != template_loci:
            counts["shape"] += 1
            continue
        ok = True
        for locus in member_loci:
            value = read_locus(member, locus)
            domain = domains.get(str(locus)) or ()
            if domain:
                if value not in domain:
                    ok = False
                    break
            elif value != read_locus(template, locus):
                ok = False
                break
        if not ok:
            counts["domain"] += 1
            continue
        key = candidate_key(member)
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
        accepted.append(dict(member))

    return PoolReport(
        accepted=tuple(accepted),
        emitted=considered,
        rejected_shape=counts["shape"],
        rejected_out_of_domain=counts["domain"],
        duplicates=counts["duplicate"],
        archive_overlap=counts["overlap"],
    )


GENERATOR_PROMPT = """You are writing the CANDIDATE GENERATOR for a black-box \
multi-objective optimizer. It calls your function every generation to draw the \
pool of configurations it will consider; you are writing the DISTRIBUTION those \
draws come from, not any particular draw.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE:
{schema}

Write ONE Python function with EXACTLY this signature:

    {contract}

Rules:
- `domains` maps every locus to its allowed values under the current sampling
  prior; sequence fields appear per position as `name[i]`. Each configuration
  you return must have the SAME SHAPE as the archive members and take every
  value from `domains` at that locus -- anything else is validated out and
  its slot falls back to a uniform random draw.
- `archive` holds configurations already measured in this run (it may be
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

Reply with ONLY one fenced Python code block and no other text."""


GENERATOR_REVISION_PROMPT = """You previously wrote this candidate generator \
for a black-box multi-objective optimizer:

```python
{source}
```

The harness ran it, validated everything it emitted, and measured what
survived. Here is what actually happened:

{feedback}

Revise the function. Read the numbers literally: rejected candidates mean
values outside `domains` or a shape that does not match the archive;
duplicates mean the sampler is collapsing; archive overlap means it keeps
re-proposing configurations already measured; no survivors means the region
it concentrates on is not competitive and the mass should move. Same rules as
before: exactly this signature

    {contract}

exactly `n` configurations, every value from `domains` at that locus, the same
shape as the archive members, randomness derived from `seed`, standard library
only ({imports}), deterministic, no I/O.

Reply with ONLY one fenced Python code block and no other text."""


def author_generator(
    complete: Callable[[str], str],
    *,
    objectives: Sequence[ObjectiveSpec],
    schema_text: str,
    attempts: int = 2,
    telemetry: Optional[AuthorTelemetry] = None,
) -> Optional[AuthoredArtifact]:
    """Ask the model to write ``propose``; accept whole or not at all."""

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["generator"]
    prompt = GENERATOR_PROMPT.format(
        goals="\n".join(f"  {s.name}: {s.goal}imise" for s in objectives),
        schema=schema_text,
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
        contract=contract.description,
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
    )
    return _author(complete, prompt, contract=contract, attempts=attempts,
                   telemetry=tel, name=f"{artifact.name}_rev")


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


def render_generation_feedback(
    telemetry: GeneratorTelemetry,
    last: Optional[PoolReport] = None,
    survivors: Sequence[Tuple[Config, Mapping[str, float]]] = (),
) -> str:
    """The measured story a generator revision needs, as text.

    Three things, all counted: how much of what it emitted the harness could
    use, how much of it was novel, and what happened to the candidates that
    were measured. Survivors are shown rather than "the best" -- ranking
    candidates across objectives would need weights nobody declared, while
    surviving truncation is the run's own weight-free verdict.
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
    if last is not None and last.emitted:
        lines.append(
            f"  most recent batch: {last.duplicate_rate:.0%} duplicates, "
            f"{last.archive_overlap_rate:.0%} already measured, "
            f"{last.novelty_rate:.0%} usable and new")
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
    ) -> None:
        if pool_factor < 1:
            raise ValueError(f"pool_factor must be at least 1, got {pool_factor}")
        if pool_size < 0:
            raise ValueError(f"pool_size must be non-negative, got {pool_size}")
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
        self.telemetry = GeneratorTelemetry()
        self.mechanism = "authored_generator"
        self.authored_by = artifact.authored_by
        self.last_report: Optional[PoolReport] = None
        self._seen: set[str] = set()
        self._survivors: List[Tuple[Config, Mapping[str, float]]] = []

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

        self._maybe_revise()
        n = self.pool_for(want)
        domains = {
            str(locus): list(locus_domain(candidate_model, locus,
                                          restriction=restriction))
            for locus in loci_of(template)
        }
        shown = [dict(config) for config in list(archive)[:self.archive_shown]]

        self.telemetry.batches += 1
        report = validate_pool(
            self._emit(shown, n, domains, seed),
            template=template, domains=domains, seen=self._seen, limit=n,
        )
        self.last_report = report
        self.telemetry.emitted += report.emitted
        self.telemetry.accepted += len(report.accepted)
        self.telemetry.rejected_shape += report.rejected_shape
        self.telemetry.rejected_out_of_domain += report.rejected_out_of_domain
        self.telemetry.duplicates += report.duplicates
        self.telemetry.archive_overlap += report.archive_overlap

        pool = [dict(config) for config in report.accepted[:n]]
        while len(pool) < n:
            pool.append(uniform_candidate(template, candidate_model, rng=rng,
                                          restriction=restriction))
            self.telemetry.filled_uniform += 1
        return pool

    def _emit(self, archive, n, domains, seed) -> Any:
        try:
            outcome = self.runtime.call(
                self.artifact, [[archive, int(n), domains, int(seed)]])
        except Exception:                    # a runtime that cannot even ship
            self.telemetry.runtime_failures += 1     # the call is a countable
            return []                                # event, not an emergency
        if not outcome.ok:
            self.telemetry.runtime_failures += 1
            return []
        [rows] = outcome.results
        if not isinstance(rows, list):
            self.telemetry.runtime_failures += 1
            return []
        return rows

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
                or tel.runtime_failures):
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
        feedback = render_generation_feedback(
            self.telemetry, self.last_report, self._survivors)
        try:
            replacement = self.revise(self.artifact, feedback)
        except Exception:                    # a revision must not kill a run
            replacement = None
        if replacement is None:
            return
        self.telemetry.revisions_accepted += 1
        self.artifact = replacement

    def note(self) -> Dict[str, Any]:
        """The per-generation history record for the last batch."""

        note = (self.last_report.as_note() if self.last_report is not None
                else PoolReport().as_note())
        note["artifact"] = f"{self.artifact.name}:{self.artifact.source_sha256[:8]}"
        note["revisions"] = self.telemetry.revisions_accepted
        return note

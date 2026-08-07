"""An operator chooser backed by a language model.

The model is asked *which parents to recombine and where to cut*. It is never
asked to write a candidate, and it could not: :class:`OperatorChoice` has no
field that can hold a configuration. That is the distinction this project's
claim rests on -- guidance over operator choice rather than artifact authoring --
and it is enforced by the type rather than by the prompt.

Everything the model sees is derived from the problem's own candidate shape and
objective declarations. Nothing here knows what a locus means.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Mapping, Sequence

from agent_evolve.policies.genetic import Locus, loci_of
from agent_evolve.session.genetic_loop import OperatorChoice

__all__ = ["ChooserTelemetry", "llm_chooser", "render_population", "parse_choices"]

Config = dict[str, Any]

#: Minimal provider surface: prompt in, text out. Deliberately not the harness
#: interface -- a chooser needs one completion, not a proposal protocol, and a
#: narrow seam is testable without a provider.
Complete = Callable[[str], str]


class ChooserTelemetry:
    """Counts what the model actually returned. Reported, never inferred.

    ``wrote_candidate`` is the one that matters: a model that answers with a
    configuration instead of operator arguments has stepped outside the claim,
    and a silent fallback to random choice would produce a null result
    indistinguishable from an honest one.
    """

    __slots__ = ("calls", "accepted", "unparseable", "malformed",
                 "wrote_candidate", "out_of_range", "errors")

    def __init__(self) -> None:
        self.calls = 0
        self.accepted = 0
        self.unparseable = 0
        self.malformed = 0
        self.wrote_candidate = 0
        self.out_of_range = 0
        self.errors = 0

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__slots__}


_PROMPT = """You are choosing recombination operations for a genetic optimizer.

OBJECTIVES (this is what the numbers mean):
{objectives}
{state}
CURRENT POPULATION -- index, then the candidate, then its measured objectives:
{population}

Each candidate has {n_loci} heritable positions, in this order:
{loci}

Budget used: {used} of {total} evaluations.

Choose {k} recombination operations. For each, name TWO parents by INDEX and a
crossover MASK of exactly {n_loci} bits: bit i = 0 takes position i from
parent_a, bit i = 1 takes it from parent_b.

CHOOSE FOR COMPLEMENTARITY, NOT RANK. Recombination only helps when the two
parents carry DIFFERENT useful material: repeatedly pairing the top-ranked
candidate, or masks that copy nearly everything from one side, spend
evaluations on near-clones. Across your {k} choices use varied parent pairs --
including mid-rank candidates whose positions differ from the leaders' -- and
prefer masks that genuinely mix both parents.

Do NOT write candidates. Choose indices and mask bits only.

Reply with ONLY a JSON list of {k} objects and no other text:
[{{"parent_a": 0, "parent_b": 2, "mask": [{example}]}}]"""


def render_population(
    population: Sequence[tuple[Config, float]],
    objectives: Sequence[Any] | None = None,
) -> str:
    """One line per member: index, candidate, rank. Compact and lossless."""

    lines = []
    for i, (config, rank) in enumerate(population):
        lines.append(f"  {i}: {json.dumps(config, sort_keys=True)}  "
                     f"(rank {rank:g}; lower is better)")
    return "\n".join(lines)


def parse_choices(
    text: str,
    *,
    n_loci: int,
    population_size: int,
    telemetry: ChooserTelemetry,
) -> list[OperatorChoice]:
    """Parse operator arguments, rejecting anything that is a candidate.

    Rejections are counted rather than silently dropped, because the number of
    rejected responses is the difference between "guidance did not help" and
    "guidance never arrived".
    """

    match = re.search(r"\[.*\]", text, re.S)
    if match is None:
        telemetry.unparseable += 1
        return []
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        telemetry.unparseable += 1
        return []
    if not isinstance(raw, list):
        telemetry.unparseable += 1
        return []

    out: list[OperatorChoice] = []
    for item in raw:
        if not isinstance(item, dict):
            telemetry.malformed += 1
            continue
        # A nested structure under any key is a candidate, not a choice. This is
        # the failure that would collapse the arm into whole-artifact rewrite.
        if any(isinstance(v, (dict,)) for v in item.values()) or "mask" not in item:
            telemetry.wrote_candidate += 1
            continue
        try:
            parent_a = int(item["parent_a"])
            parent_b = int(item["parent_b"])
            mask = tuple(bool(int(bit)) for bit in item["mask"])
        except (KeyError, TypeError, ValueError):
            telemetry.malformed += 1
            continue
        if not (0 <= parent_a < population_size and 0 <= parent_b < population_size):
            telemetry.out_of_range += 1
            continue
        if len(mask) != n_loci:
            telemetry.out_of_range += 1
            continue
        out.append(OperatorChoice(parent_a=parent_a, parent_b=parent_b, mask=mask))
    telemetry.accepted += len(out)
    return out


def llm_chooser(
    complete: Complete,
    *,
    objectives: Sequence[Any],
    budget: int,
    telemetry: ChooserTelemetry | None = None,
    on_shortfall: Callable[[int, int], None] | None = None,
):
    """Build an :class:`OperatorChooser` that asks *complete* for choices.

    When the model returns fewer usable choices than asked for, the caller is
    told through *on_shortfall* and the loop fills the remainder itself. A run
    that quietly substitutes random choices for model choices is a run whose
    result cannot be attributed.
    """

    stats = telemetry if telemetry is not None else ChooserTelemetry()
    spent = {"n": 0}

    def choose(population: Sequence[tuple[Config, float]], count: int,
               state: Any = None):
        if not population:
            return []
        loci = loci_of(population[0][0])
        # An absent state renders to nothing at all, so the score-only baseline
        # is byte-identical to the prompt this chooser sent before the search
        # state existed -- an ablation must not carry an empty heading.
        block = ""
        if state is not None:
            rendered = state.render([spec.name for spec in objectives])
            if rendered:
                block = "\n" + rendered + "\n"
        prompt = _PROMPT.format(
            objectives="\n".join(
                f"  {spec.name}: {spec.goal}imise" for spec in objectives
            ),
            state=block,
            population=render_population(population, objectives),
            n_loci=len(loci),
            loci="  " + ", ".join(str(locus) for locus in loci),
            used=spent["n"],
            total=budget,
            k=count,
            example=",".join("0" for _ in loci),
        )
        stats.calls += 1
        try:
            text = complete(prompt)
        except Exception:
            stats.errors += 1
            return []
        choices = parse_choices(
            text, n_loci=len(loci), population_size=len(population),
            telemetry=stats,
        )
        spent["n"] += count
        if len(choices) < count and on_shortfall is not None:
            on_shortfall(len(choices), count)
        return choices

    choose.telemetry = stats          # type: ignore[attr-defined]
    choose.mechanism = "chooser"      # type: ignore[attr-defined]
    choose.authored_by = "llm"        # type: ignore[attr-defined]
    return choose

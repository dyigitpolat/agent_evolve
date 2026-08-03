"""What the operator chooser is allowed to reason over.

The chooser's job is to pick parents, cut points and loci. Until now it saw only
a population listing and a rank, which is a thin question -- and a thin question
is why a small model and a large one would answer it alike. The storyline asks
for guidance that reasons over *measurements*; this module is those measurements.

Four channels, each independently switchable. They are separable on purpose:
stacking reflective signals in a closed loop is reported to raise the median and
inflate the variance together, in ways not predictable from the components
(MAGE, arXiv:2607.11944). With four stacked and no switches we would not learn
which one earned its place.

Nothing here knows what a locus means. Every channel is derived from the
problem's own candidate shape and objective declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from agent_evolve.policies.genetic import Locus, loci_of, read_locus

__all__ = [
    "ValueStat",
    "StateChannels",
    "locus_table",
    "render_locus_table",
    "render_history",
    "SearchState",
]

Config = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ValueStat:
    """How one value at one locus has performed, per objective.

    Objectives are kept SEPARATE rather than scalarised. A scalarisation needs
    weights nobody declared, and an undeclared weight is exactly the kind of
    hidden choice that has silently decided results in this project.
    """

    value: Any
    count: int
    mean: dict[str, float]
    best: dict[str, float]


@dataclass(frozen=True, slots=True)
class StateChannels:
    """Which context channels are enabled. All four on is the full design."""

    locus_measurements: bool = True
    side_information: bool = True
    recent_history: bool = True
    distilled_rules: bool = True

    @classmethod
    def only(cls, name: str) -> "StateChannels":
        """Everything off except *name* -- the single-channel ablation arm."""
        fields = {f: False for f in
                  ("locus_measurements", "side_information",
                   "recent_history", "distilled_rules")}
        if name not in fields:
            raise ValueError(f"unknown channel {name!r}; expected one of {sorted(fields)}")
        fields[name] = True
        return cls(**fields)

    @classmethod
    def none(cls) -> "StateChannels":
        """The score-only baseline: exactly what the chooser saw before."""
        return cls(False, False, False, False)


def locus_table(
    evaluated: Sequence[tuple[Config, Mapping[str, float]]],
    objective_names: Sequence[str],
) -> dict[Locus, list[ValueStat]]:
    """Per locus, how each value tried there has performed.

    This is the channel that makes the chooser's decision deep enough for model
    capability to matter: from it a reader can infer which loci are saturated,
    which carry the current gain, and which are barely explored. A reader who
    cannot make those inferences will choose more arbitrarily, which is the
    effect the capability ladder is meant to detect.
    """

    if not evaluated:
        return {}
    table: dict[Locus, dict[Any, list[Mapping[str, float]]]] = {}
    for config, objectives in evaluated:
        for locus in loci_of(config):
            value = read_locus(config, locus)
            key = value if isinstance(value, (str, int, float, bool)) else repr(value)
            table.setdefault(locus, {}).setdefault(key, []).append(objectives)

    out: dict[Locus, list[ValueStat]] = {}
    for locus, by_value in table.items():
        stats: list[ValueStat] = []
        for value, rows in by_value.items():
            mean = {n: sum(float(r[n]) for r in rows) / len(rows)
                    for n in objective_names if rows and n in rows[0]}
            best = {n: min(float(r[n]) for r in rows)
                    for n in objective_names if rows and n in rows[0]}
            stats.append(ValueStat(value=value, count=len(rows), mean=mean, best=best))
        stats.sort(key=lambda s: (-s.count, str(s.value)))
        out[locus] = stats
    return out


def render_locus_table(
    table: Mapping[Locus, Sequence[ValueStat]],
    objective_names: Sequence[str],
    *,
    max_values_per_locus: int = 6,
) -> str:
    """Render the table, marking saturation and under-exploration explicitly.

    The annotations are computed, not adjectives: a locus is *saturated* when one
    value holds every observation, and *unexplored* when it has been measured
    once. Stating them saves every reader from re-deriving the same arithmetic,
    and states plainly what a weaker reader might miss.
    """

    if not table:
        return "  (nothing evaluated yet)"
    lines: list[str] = []
    for locus in sorted(table, key=lambda lc: (lc.field, lc.index if lc.index is not None else -1)):
        stats = list(table[locus])
        total = sum(s.count for s in stats)
        shown = stats[:max_values_per_locus]
        parts = []
        for s in shown:
            metrics = " ".join(
                f"{n}~{s.mean[n]:.4g}/best {s.best[n]:.4g}"
                for n in objective_names if n in s.mean
            )
            parts.append(f"{s.value}(n={s.count} {metrics})")
        note = ""
        if len(stats) == 1 and total > 1:
            note = "   [SATURATED: one value holds every observation]"
        elif total <= 1:
            note = "   [barely explored]"
        elided = len(stats) - len(shown)
        tail = f" (+{elided} more values)" if elided > 0 else ""
        lines.append(f"  {locus}: " + "  ".join(parts) + tail + note)
    return "\n".join(lines)


def render_history(history: Sequence[Mapping[str, Any]], *, keep: int = 2) -> str:
    """The last *keep* generations, as what changed and what it bought."""

    if not history:
        return "  (no generations yet)"
    lines = []
    for record in list(history)[-keep:]:
        gen = record.get("gen", "?")
        made = record.get("valid_count", 0)
        best = record.get("best_after")
        filled = record.get("choices_filled_at_random", 0)
        line = f"  generation {gen}: {made} offspring evaluated"
        if best is not None:
            line += f", best after = {best}"
        if filled:
            line += f"  [{filled} choices were filled at random, not chosen]"
        lines.append(line)
    return "\n".join(lines)


@dataclass(slots=True)
class SearchState:
    """Everything the chooser may reason over, assembled once per generation."""

    channels: StateChannels = field(default_factory=StateChannels)
    evaluated: list[tuple[Config, Mapping[str, float]]] = field(default_factory=list)
    history: list[Mapping[str, Any]] = field(default_factory=list)
    side_information: list[str] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def render(self, objective_names: Sequence[str]) -> str:
        """The context block. Disabled channels contribute nothing at all.

        A disabled channel is absent rather than empty: an empty section still
        tells the reader the channel exists, which is a difference an ablation
        should not carry.
        """

        blocks: list[str] = []
        if self.channels.locus_measurements:
            blocks.append(
                "PER-LOCUS MEASUREMENTS (what has been tried at each position, "
                "and what it scored):\n"
                + render_locus_table(
                    locus_table(self.evaluated, objective_names), objective_names
                )
            )
        if self.channels.side_information and self.side_information:
            blocks.append("EVALUATOR DIAGNOSTICS:\n  " +
                          "\n  ".join(self.side_information[-8:]))
        if self.channels.recent_history:
            blocks.append("RECENT GENERATIONS:\n" + render_history(self.history))
        if self.channels.distilled_rules and self.rules:
            blocks.append("WHAT HAS HELD SO FAR ON THIS PROBLEM:\n  " +
                          "\n  ".join(self.rules[-6:]))
        return "\n\n".join(blocks)

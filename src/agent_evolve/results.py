"""Result containers and Pareto-front utilities for agent_evolve."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from agent_evolve.problem import ObjectiveSpec

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class Candidate(Generic[ConfigT]):
    """A single evaluated configuration."""

    configuration: ConfigT
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult(Generic[ConfigT]):
    """Aggregated output of an optimisation run.

    Attributes
    ----------
    objectives : objectives used during the run.
    best : the single recommended candidate.
    pareto_front : non-dominated set.
    all_candidates : every evaluated candidate across all generations.
    history : per-generation summary dicts.
    """

    objectives: Sequence[ObjectiveSpec]
    best: Candidate[ConfigT]
    pareto_front: List[Candidate[ConfigT]] = field(default_factory=list)
    all_candidates: List[Candidate[ConfigT]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------
# Pareto dominance
# ------------------------------------------------------------------

def dominates(
    a: Dict[str, float],
    b: Dict[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Return *True* if objective vector *a* Pareto-dominates *b*."""
    all_geq = True
    any_better = False
    for spec in objectives:
        va = a.get(spec.name, 0.0)
        vb = b.get(spec.name, 0.0)
        if spec.goal == "max":
            if va < vb:
                all_geq = False
            elif va > vb:
                any_better = True
        else:
            if va > vb:
                all_geq = False
            elif va < vb:
                any_better = True
    return all_geq and any_better


def compute_pareto_front(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[Candidate[ConfigT]]:
    """Return the non-dominated subset of *candidates*."""
    if not candidates:
        return []
    front: List[Candidate[ConfigT]] = []
    for i, c in enumerate(candidates):
        if not any(
            dominates(other.objectives, c.objectives, objectives)
            for j, other in enumerate(candidates)
            if j != i
        ):
            front.append(c)
    return front


# ------------------------------------------------------------------
# Best-candidate selection
# ------------------------------------------------------------------

def select_best_candidate(
    pareto: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
    priority_order: Optional[List[str]] = None,
) -> Optional[Candidate[ConfigT]]:
    """Lexicographic selection from the Pareto front.

    Default priority: maximise objectives first, then minimise objectives.
    """
    if not pareto:
        return None
    if priority_order is None:
        max_objs = [s for s in objectives if s.goal == "max"]
        min_objs = [s for s in objectives if s.goal == "min"]
        priority_order = [s.name for s in max_objs] + [s.name for s in min_objs]
    obj_map = {s.name: s for s in objectives}

    def _key(c: Candidate[ConfigT]) -> Tuple[float, ...]:
        parts: List[float] = []
        for name in priority_order:
            spec = obj_map.get(name)
            val = c.objectives.get(name, 0.0)
            parts.append(-val if spec and spec.goal == "max" else val)
        return tuple(parts)

    return min(pareto, key=_key)


# ------------------------------------------------------------------
# Minimax-rank selection
# ------------------------------------------------------------------

def _rank_candidates(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[List[int]]:
    """``ranks[i][j]`` = 1-based dense rank of candidate *i* on objective *j*."""
    n = len(candidates)
    ranks: List[List[int]] = [[0] * len(objectives) for _ in range(n)]
    for j, spec in enumerate(objectives):
        values = [float(c.objectives.get(spec.name, 0.0)) for c in candidates]
        reverse = spec.goal == "max"
        order = sorted(range(n), key=lambda i: values[i], reverse=reverse)
        cur = 1
        for pos, idx in enumerate(order):
            if pos > 0 and values[order[pos]] != values[order[pos - 1]]:
                cur = pos + 1
            ranks[idx][j] = cur
    return ranks


def select_minimax_rank(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Candidate[ConfigT]]:
    """Pick the most balanced candidate (smallest worst-rank across objectives).

    Ties broken by smallest sum of ranks.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    ranks = _rank_candidates(candidates, objectives)
    worst = [max(r) for r in ranks]
    min_worst = min(worst)
    tied = [i for i in range(len(candidates)) if worst[i] == min_worst]
    if len(tied) == 1:
        return candidates[tied[0]]
    best_idx = min(tied, key=lambda i: sum(ranks[i]))
    return candidates[best_idx]

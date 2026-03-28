"""Internal helpers used by evolve.kedi's embedded Python blocks.

Public API users should not import from this module directly.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent_evolve.problem import ObjectiveSpec
from agent_evolve.results import (
    Candidate,
    SearchResult,
    compute_pareto_front,
    dominates,
    select_best_candidate,
)


# ------------------------------------------------------------------
# Internal candidate tracking
# ------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Intermediate evaluation record (richer than public Candidate)."""

    configuration: Dict[str, Any]
    objectives: Dict[str, float]
    is_valid: bool
    error_message: Optional[str] = None
    insight: str = ""


# ------------------------------------------------------------------
# Evaluate a batch of candidates against a Problem
# ------------------------------------------------------------------

INVALID_PENALTY: float = 1e18


def evaluate_batch(
    problem: Any,
    candidates: List[Dict[str, Any]],
    objectives: Sequence[ObjectiveSpec],
    *,
    verbose: bool = True,
    log_fn: Callable[[str], None] = lambda m: print(m, flush=True),
) -> Tuple[List[CandidateResult], List[CandidateResult]]:
    """Evaluate *candidates* against *problem*.

    Returns ``(valid_results, failed_results)``.
    """
    valid: List[CandidateResult] = []
    failed: List[CandidateResult] = []
    has_validate = hasattr(problem, "validate")

    for idx, config in enumerate(candidates):
        try:
            if verbose:
                log_fn(f"    Candidate {idx + 1}: {prettify_configuration(config)[:200]}...")

            if has_validate and not problem.validate(config):
                _append_failure(failed, config, objectives, "Validation failed", verbose, log_fn)
                continue

            obj = problem.evaluate(config)
            if verbose:
                log_fn(f"    -> VALID: {obj}")
            valid.append(CandidateResult(configuration=config, objectives=obj, is_valid=True))

        except Exception as exc:
            _append_failure(failed, config, objectives, str(exc), verbose, log_fn)

    return valid, failed


def _append_failure(
    bucket: List[CandidateResult],
    config: Dict[str, Any],
    objectives: Sequence[ObjectiveSpec],
    message: str,
    verbose: bool,
    log_fn: Callable[[str], None],
) -> None:
    if verbose:
        log_fn(f"    -> FAILED: {message}")
    bucket.append(
        CandidateResult(
            configuration=config,
            objectives={
                s.name: (0.0 if s.goal == "max" else INVALID_PENALTY) for s in objectives
            },
            is_valid=False,
            error_message=message,
        )
    )


# ------------------------------------------------------------------
# Formatting helpers (consumed by LLM prompts)
# ------------------------------------------------------------------

def prettify_configuration(config: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(config, indent=indent, sort_keys=True)


def prettify_objectives(objectives: Sequence[ObjectiveSpec]) -> str:
    lines = ["OBJECTIVES:", "=" * 60]
    for spec in objectives:
        desc = "higher is better" if spec.goal == "max" else "lower is better"
        lines.append(f"  - {spec.name}: {desc}")
    return "\n".join(lines)


def prettify_results(
    results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> str:
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"--- Candidate {i} ---")
        lines.append(f"Configuration: {prettify_configuration(r.configuration)}")
        if r.is_valid:
            parts = []
            for spec in objectives:
                val = r.objectives.get(spec.name, 0.0)
                arrow = "\u2191" if spec.goal == "max" else "\u2193"
                parts.append(f"{spec.name}={val:.4f}{arrow}")
            lines.append(f"Objectives: {', '.join(parts)}")
        else:
            lines.append("Status: INVALID")
            if r.error_message:
                lines.append(f"Error: {r.error_message}")
        if r.insight:
            lines.append(f"Insight: {r.insight}")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Search-space description for LLM context
# ------------------------------------------------------------------

def format_search_space_description(
    objectives: Sequence[ObjectiveSpec],
    *,
    config_schema: Optional[Dict[str, Any]] = None,
    example_config: Optional[Dict[str, Any]] = None,
    constraints: Optional[str] = None,
    problem_description: Optional[str] = None,
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("MULTI-OBJECTIVE OPTIMIZATION PROBLEM")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVES:")
    for spec in objectives:
        desc = "MAXIMIZE (higher is better)" if spec.goal == "max" else "MINIMIZE (lower is better)"
        lines.append(f"  \u2022 {spec.name}: {desc}")
    lines.append("")

    if problem_description:
        lines.append("PROBLEM DESCRIPTION:")
        lines.append(problem_description)
        lines.append("")

    if config_schema:
        lines.append("CONFIGURATION SCHEMA:")
        lines.append(prettify_configuration(config_schema))
        lines.append("")

    if example_config:
        lines.append("EXAMPLE CONFIGURATION:")
        lines.append(prettify_configuration(example_config))
        lines.append("")

    if constraints:
        lines.append("CONSTRAINTS:")
        lines.append(constraints)
        lines.append("")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Performance statistics
# ------------------------------------------------------------------

def compute_performance_stats(
    valid_results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Dict[str, Any]]:
    """Compute best/worst per objective and top Pareto candidates."""
    if not valid_results:
        return None

    stats: Dict[str, Any] = {}

    for spec in objectives:
        key = spec.name
        if spec.goal == "max":
            best = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
            worst = min(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        else:
            best = min(valid_results, key=lambda r: r.objectives.get(key, float("inf")))
            worst = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        stats[f"best_{key}"] = best
        stats[f"worst_{key}"] = worst

    candidates = [result_to_candidate(r) for r in valid_results]
    pareto_candidates = compute_pareto_front(candidates, objectives)
    pareto_results = [candidate_to_result(c) for c in pareto_candidates]

    def _ranking_score(cr: CandidateResult) -> float:
        total_rank = 0
        for spec in objectives:
            val = cr.objectives.get(spec.name, 0.0)
            if spec.goal == "max":
                rank = sum(1 for o in pareto_results if o.objectives.get(spec.name, 0.0) > val) + 1
            else:
                rank = sum(1 for o in pareto_results if o.objectives.get(spec.name, float("inf")) < val) + 1
            total_rank += rank
        return 1.0 / total_rank if total_rank > 0 else 0.0

    sorted_pareto = sorted(pareto_results, key=_ranking_score, reverse=True)
    stats["top_3_pareto"] = sorted_pareto[:3]
    stats["pareto_front"] = pareto_results
    stats["pareto_size"] = len(pareto_results)

    return stats


# ------------------------------------------------------------------
# Failure sampling for constraint learning
# ------------------------------------------------------------------

def sample_failed_for_constraint(
    latest_failed: List[CandidateResult],
    all_previous_failed: List[CandidateResult],
    max_examples: int,
) -> List[CandidateResult]:
    """Sample failures for constraint-instruction generation.

    Always includes latest failures; fills remaining slots with random
    previous failures.
    """
    sampled = list(latest_failed)
    if len(sampled) >= max_examples:
        return sampled[:max_examples]

    remaining = max_examples - len(sampled)
    latest_ids = {id(r) for r in latest_failed}
    previous = [r for r in all_previous_failed if id(r) not in latest_ids]
    if previous and remaining > 0:
        sampled.extend(random.sample(previous, min(remaining, len(previous))))
    return sampled


# ------------------------------------------------------------------
# Conversions between CandidateResult and public Candidate
# ------------------------------------------------------------------

def result_to_candidate(
    result: CandidateResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> Candidate[Dict[str, Any]]:
    return Candidate(
        configuration=result.configuration,
        objectives=result.objectives,
        metadata=metadata or {"is_pareto": False},
    )


def candidate_to_result(candidate: Candidate[Dict[str, Any]]) -> CandidateResult:
    return CandidateResult(
        configuration=candidate.configuration,
        objectives=candidate.objectives,
        is_valid=True,
    )


# ------------------------------------------------------------------
# Parse LLM candidate output
# ------------------------------------------------------------------

def parse_candidates(
    candidates: Any,
    expected_count: int,
    log_fn: Callable[[str], None] = lambda m: print(m, flush=True),
) -> List[Dict[str, Any]]:
    """Normalise LLM output into a list of configuration dicts."""
    if not isinstance(candidates, list):
        log_fn(f"Warning: LLM returned non-list candidates: {type(candidates)}")
        return []

    parsed: List[Dict[str, Any]] = []
    for c in candidates:
        if isinstance(c, dict):
            parsed.append(c)
        elif isinstance(c, str):
            try:
                parsed.append(json.loads(c))
            except Exception:
                log_fn(f"Warning: Could not parse candidate string: {c[:100]}")

    if len(parsed) != expected_count:
        log_fn(f"Warning: Expected {expected_count} candidates, got {len(parsed)}")
    return parsed


# ------------------------------------------------------------------
# Build final SearchResult from internal bookkeeping
# ------------------------------------------------------------------

def build_search_result(
    all_valid: List[CandidateResult],
    all_candidates_meta: List[Tuple[CandidateResult, Dict[str, Any]]],
    objectives: Sequence[ObjectiveSpec],
    history: List[Dict[str, Any]],
) -> SearchResult[Dict[str, Any]]:
    """Assemble the public SearchResult from internal data."""
    pareto_results = compute_pareto_front(
        [result_to_candidate(r) for r in all_valid], objectives
    )
    pareto_configs = {prettify_configuration(c.configuration) for c in pareto_results}

    all_candidates: List[Candidate[Dict[str, Any]]] = []
    for cr, meta in all_candidates_meta:
        c_key = prettify_configuration(cr.configuration)
        meta_copy = dict(meta)
        if c_key in pareto_configs:
            meta_copy["is_pareto"] = True
        all_candidates.append(result_to_candidate(cr, meta_copy))

    pareto_list = [
        result_to_candidate(candidate_to_result(c), {"is_pareto": True})
        for c in pareto_results
    ]

    best_candidate = select_best_candidate(pareto_results, objectives)
    if best_candidate is None:
        best_candidate = Candidate(configuration={}, objectives={}, metadata={})

    return SearchResult(
        objectives=list(objectives),
        best=best_candidate,
        pareto_front=pareto_list,
        all_candidates=all_candidates,
        history=history,
    )

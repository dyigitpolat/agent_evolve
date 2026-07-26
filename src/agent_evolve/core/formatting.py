"""Pure formatting/parsing helpers shared by the loop and the LLM prompts.

Everything here is a pure function of its arguments (no I/O, no globals).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.core.results import Candidate, objective_value


@dataclass
class CandidateResult:
    """Intermediate evaluation record (richer than the public :class:`Candidate`)."""

    configuration: Dict[str, Any]
    objectives: Dict[str, float]
    is_valid: bool
    error_message: Optional[str] = None
    failure_phase: Optional[str] = None
    #: True iff ``Problem.evaluate`` was actually invoked for this candidate.
    evaluation_attempted: bool = False
    insight: str = ""
    #: Original element from the LLM ``candidates`` list (before/while parsing).
    raw_llm_element: Optional[Any] = None


# ------------------------------------------------------------------
# Prettifiers (consumed by LLM prompts)
# ------------------------------------------------------------------

def prettify_configuration(config: Dict[str, Any], indent: int = 2) -> str:
    return json.dumps(config, indent=indent, sort_keys=True, default=str)


def dump_raw_llm_element(obj: Any, *, max_len: int = 12_000) -> str:
    """Serialize a raw LLM list element for logs and failure prompts."""
    if obj is None:
        return "(none)"
    try:
        s = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, {len(s)} chars total]"
    return s


def prettify_objectives(objectives: Sequence[ObjectiveSpec]) -> str:
    lines = ["OBJECTIVES:", "=" * 60]
    for spec in objectives:
        desc = "higher is better" if spec.goal == "max" else "lower is better"
        lines.append(f"  - {spec.name}: {desc}")
    return "\n".join(lines)


def prettify_results(
    results: Sequence[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
    render: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> str:
    """Format candidate results for LLM prompts.

    ``render`` optionally produces a compact one-line view of each configuration
    (e.g. a problem-specific summary); when absent, pretty JSON is used.
    """
    lines: List[str] = []
    for i, r in enumerate(results, 1):
        lines.append(f"--- Candidate {i} ---")
        config_str = render(r.configuration) if render is not None else prettify_configuration(r.configuration)
        lines.append(f"Configuration: {config_str}")
        if getattr(r, "raw_llm_element", None) is not None:
            lines.append(
                "Raw LLM element (exact item from the model's candidates list): "
                + dump_raw_llm_element(r.raw_llm_element)
            )
        if r.is_valid:
            parts = []
            for spec in objectives:
                val = objective_value(r.objectives, spec.name)
                arrow = "\u2191" if spec.goal == "max" else "\u2193"
                parts.append(f"{spec.name}={val:.4f}{arrow}")
            lines.append(f"Objectives: {', '.join(parts)}")
        else:
            lines.append("Status: INVALID")
            if r.failure_phase:
                lines.append(f"Failure Phase: {r.failure_phase}")
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
# Conversions between CandidateResult and the public Candidate
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
        evaluation_attempted=True,
    )


# ------------------------------------------------------------------
# Parse LLM candidate output
# ------------------------------------------------------------------

def parse_llm_json_array(s: str) -> List[Any]:
    """Parse a JSON array (or single object) from an LLM ``str`` field.

    The single-JSON-string output contract avoids the ``[{}, {}, ...]`` degradation
    that ``list[dict]`` structured outputs are prone to. Strips optional markdown fences.
    """
    s = (s or "").strip()
    if not s:
        raise ValueError("Empty candidates JSON string")
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    data = json.loads(s)
    if isinstance(data, dict):
        inner = data.get("candidates")
        if isinstance(inner, list):
            return inner
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"Candidates JSON must be an array or object, got {type(data).__name__}")


def parse_candidates(
    candidates: Any,
    expected_count: int,
    log_fn: Callable[[str], None] = lambda m: None,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Normalise LLM output into configuration dicts.

    Returns ``(parsed_configs, raw_elements)`` with one raw element per input list
    item (same order), so failures can show what the model actually returned even
    when the parsed dict is ``{}`` or wrong.
    """
    if isinstance(candidates, str):
        try:
            candidates = parse_llm_json_array(candidates)
        except Exception as exc:
            log_fn(f"Warning: could not parse candidates as JSON array string: {exc}")
            return [], []

    if isinstance(candidates, dict):
        inner = candidates.get("candidates")
        if isinstance(inner, list):
            candidates = inner
        else:
            log_fn(
                f"Warning: LLM returned a dict without a 'candidates' list "
                f"(keys: {list(candidates.keys())})."
            )
            return [], []

    if not isinstance(candidates, list):
        log_fn(f"Warning: LLM returned non-list candidates: {type(candidates)}")
        return [], []

    parsed: List[Dict[str, Any]] = []
    raw_elements: List[Any] = []
    for c in candidates:
        raw_elements.append(c)
        if isinstance(c, BaseModel):
            parsed.append(c.model_dump())
        elif isinstance(c, dict):
            parsed.append(c)
        elif isinstance(c, str):
            try:
                parsed.append(json.loads(c))
            except Exception:
                log_fn(f"Warning: Could not parse candidate string: {c[:100]}")
                parsed.append({})
        else:
            log_fn(
                f"Warning: candidate element has unexpected type {type(c).__name__}"
            )
            parsed.append({})

    if len(parsed) != expected_count:
        log_fn(f"Warning: Expected {expected_count} candidates, got {len(parsed)}")
    return parsed, raw_elements

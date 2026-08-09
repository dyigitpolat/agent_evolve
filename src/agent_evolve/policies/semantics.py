"""What the model is told things MEAN. One renderer, shared by every prompt.

The program's own trace-first findings (I7) showed information-channel
defects masquerading as capability failures: an objective traded against but
never rendered, grids shown as bare number lists. The structural channel
(names, domains, directions) was always faithful; the SEMANTIC channel --
what a parameter is, what an objective measures, what the evaluator does --
had no carrier at all. This module is that carrier.

Sources, all harvested rather than invented, all optional, all already
writable by a problem author today:

- ``search_space_description()`` (existing optional method), else the
  problem class docstring, else the candidate model's docstring -- the
  domain in prose;
- ``evaluate``'s docstring -- what an evaluation actually runs;
- ``ObjectiveSpec.description`` -- what each objective measures;
- pydantic ``Field(description=...)`` -- what each parameter means, read
  from the model's own JSON schema (these were previously harvested by
  nothing and silently dropped).

Every helper degrades to the structural rendering when a source is absent:
a missing description never breaks a prompt, it just starves it.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

from agent_evolve.policies.genetic import Locus, locus_domain

__all__ = ["objective_lines", "parameter_lines", "domain_card"]

_MAX_VALUES_SHOWN = 12
_MAX_PROSE = 900


def _first_paragraph(text: Optional[str]) -> str:
    if not text:
        return ""
    paragraph = text.strip().split("\n\n")[0]
    collapsed = " ".join(paragraph.split())
    return collapsed[:_MAX_PROSE]


def objective_lines(specs: Sequence[Any]) -> List[str]:
    """One line per objective: name, direction, and its meaning when given."""

    out = []
    for spec in specs:
        line = f"{spec.name} ({spec.goal})"
        description = _first_paragraph(getattr(spec, "description", ""))
        if description:
            line += f" -- {description}"
        out.append(line)
    return out


def _rendered_domain(domain: tuple) -> str:
    if len(domain) <= _MAX_VALUES_SHOWN:
        return f"one of {list(domain)}"
    head = list(domain[:_MAX_VALUES_SHOWN - 2])
    return (f"one of {head + ['...'] + [domain[-1]]} "
            f"({len(domain)} levels)")


def parameter_lines(
    candidate_model: Any,
    *,
    fields: Optional[Iterable[str]] = None,
) -> List[str]:
    """One line per field: domain, shape, and the Field description when given."""

    try:
        schema = candidate_model.model_json_schema()
    except Exception:
        return []
    properties = schema.get("properties", {}) or {}
    wanted = list(fields) if fields is not None else list(properties)
    out = []
    for name in wanted:
        node = properties.get(name, {})
        if not isinstance(node, dict):
            node = {}
        domain = locus_domain(candidate_model, Locus(name))
        is_sequence = node.get("type") == "array"
        if domain:
            shape = ("sequence field, each position "
                     if is_sequence else "") + _rendered_domain(domain)
        elif is_sequence:
            shape = "sequence field"
        else:
            shape = "unconstrained by the schema"
        line = f"{name}: {shape}"
        description = _first_paragraph(node.get("description"))
        if description:
            line += f" -- {description}"
        out.append(line)
    return out


def domain_card(problem: Any) -> str:
    """Everything the model should be told about WHAT it is optimizing.

    Renders only the sections that have content; a problem that documents
    nothing yields the structural card (objectives + parameters), which is
    exactly what prompts carried before this module existed.
    """

    blocks: List[str] = []

    prose = ""
    describe = getattr(problem, "search_space_description", None)
    if callable(describe):
        try:
            prose = _first_paragraph(describe()) or ""
            # search_space_description is often multi-paragraph and entirely
            # load-bearing; keep it whole rather than first-paragraph only.
            prose = str(describe()).strip()[:_MAX_PROSE]
        except Exception:
            prose = ""
    if not prose:
        prose = _first_paragraph(getattr(type(problem), "__doc__", ""))
    if not prose:
        candidate_model = getattr(problem, "candidate_model", None)
        prose = _first_paragraph(getattr(candidate_model, "__doc__", ""))
    if prose:
        blocks.append(f"PROBLEM:\n{prose}")

    evaluate = getattr(problem, "evaluate", None)
    evaluation = _first_paragraph(getattr(evaluate, "__doc__", ""))
    if evaluation:
        blocks.append(f"EVALUATION:\n{evaluation}")

    objectives = list(getattr(problem, "objectives", ()) or ())
    if objectives:
        blocks.append("OBJECTIVES:\n" + "\n".join(
            f"  {line}" for line in objective_lines(objectives)))

    candidate_model = getattr(problem, "candidate_model", None)
    if candidate_model is not None:
        lines = parameter_lines(candidate_model)
        if lines:
            blocks.append("PARAMETERS:\n" + "\n".join(
                f"  {line}" for line in lines))

    return "\n\n".join(blocks)

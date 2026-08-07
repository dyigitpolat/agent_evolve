"""The model writes variation operators: moves, not candidates.

A hand-rolled operator zoo encodes what its author guessed about structure
in general. The model is asked once, from the schema and objective meanings,
to write up to three ``vary`` functions encoding what THIS domain's
structure rewards -- co-moving coupled parameters, preserving named motifs,
reordering adjacent actions. Each authored operator becomes one arm in the
measured portfolio: it earns offspring slots through survival credit, its
children pass the material check or its slot falls back to the classical
arm, and the preregistered retirement rule ends arms that never produce a
survivor. The model authors the move; the run decides what it was worth.

Extraction and gating mirror ``llm_surrogate``: fenced blocks only, the
worker's own import gate applied at authoring time, every rejection counted.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

from agent_evolve.core.authored import CONTRACTS, AuthoredArtifact, authored_artifact
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.infrastructure.authored_worker import (
    ALLOWED_IMPORTS,
    _forbidden_import,
)
from agent_evolve.policies.llm_surrogate import AuthorTelemetry

__all__ = ["author_operators", "OPERATOR_PROMPT"]


OPERATOR_PROMPT = """You are writing VARIATION OPERATORS for a genetic \
optimizer over a structured configuration space. An operator constructs one \
child from two parents; good operators exploit what the parameters MEAN --
co-move parameters that are physically coupled, preserve fragments that only \
work together, reorder adjacent actions, move along known trade-offs.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE:
{schema}

Write up to {max_operators} DIFFERENT operators. Each must be one Python \
function with EXACTLY this signature:

    {contract}

Rules:
- `loci` lists the heritable positions in order (sequence fields appear as
  `name[i]`); `domains` maps each locus to its allowed values under the
  current sampling prior. Every value you place must come from a parent at
  that locus or from `domains` -- anything else is rejected and wastes the
  slot.
- Derive all randomness from `seed` (e.g. `random.Random(seed)`), so a child
  is reproducible.
- Standard library only; imports limited to: {imports}.
- Each operator in its OWN fenced Python block, each defining `vary`.
- Do not write candidates, wrappers, or commentary between blocks.

Reply with ONLY the fenced code blocks."""


_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def author_operators(
    complete: Callable[[str], str],
    *,
    objectives: Sequence[ObjectiveSpec],
    schema_text: str,
    max_operators: int = 3,
    attempts: int = 2,
    telemetry: Optional[AuthorTelemetry] = None,
) -> List[AuthoredArtifact]:
    """Ask for up to *max_operators* ``vary`` functions; gate each block whole."""

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["operator"]
    prompt = OPERATOR_PROMPT.format(
        goals="\n".join(f"  {s.name}: {s.goal}imise" for s in objectives),
        schema=schema_text,
        max_operators=max_operators,
        contract=contract.description,
        imports=", ".join(sorted(ALLOWED_IMPORTS)),
    )
    for _attempt in range(max(1, attempts)):
        tel.calls += 1
        try:
            text = complete(prompt)
        except Exception:
            tel.errors += 1
            continue
        blocks = _FENCE.findall(text)
        if not blocks:
            tel.no_code_block += 1
            continue
        artifacts: List[AuthoredArtifact] = []
        for index, source in enumerate(blocks[:max_operators]):
            try:
                forbidden = _forbidden_import(source)
            except SyntaxError:
                tel.unparseable += 1
                continue
            if forbidden is not None:
                tel.forbidden_import += 1
                continue
            if not re.search(rf"^def {contract.entry_point}\(", source, re.M):
                tel.wrong_entry_point += 1
                continue
            tel.accepted += 1
            tel.sources.append(source)
            artifacts.append(authored_artifact(
                "operator", source,
                name=f"llm_op{index + 1}", authored_by="llm",
            ))
        if artifacts:
            return artifacts
    return []

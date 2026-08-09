"""The model writes the surrogate: structure from meaning, fit from data.

This is the first mechanism in the package whose task a rule cannot do. A
rule can FIT anything -- the shipped additive and kNN surrogates are exactly
that -- but it cannot decide, from what the parameters MEAN, that gain-
bandwidth divides by a compensation capacitance or that two pass-orderings
are interchangeable. The model is asked once, before any evaluation, to
write ``fit_predict`` source encoding that structure; the harness then fits
and validates it on real data every generation, against the rule surrogates,
under the same gate. Authorship is the model's; arbitration never is.

Rejection semantics follow ``llm_prior``: the reply is used whole or not at
all. No fenced block, a block that defines the wrong name, a forbidden
import, or unparseable source each count and yield nothing -- the rule
surrogates then carry the run, so failure costs the caller a completion, not
a result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from agent_evolve.core.authored import CONTRACTS, AuthoredArtifact, authored_artifact
from agent_evolve.core.problem import ObjectiveSpec
# The worker applies this same gate before executing; sharing the function is
# what keeps "rejected at authoring time" and "rejected at run time" one rule.
from agent_evolve.infrastructure.authored_worker import (
    ALLOWED_IMPORTS,
    _forbidden_import,
)
from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
from agent_evolve.policies.surrogate import Predict, SurrogateBuilder

__all__ = [
    "AuthorTelemetry",
    "author_surrogate",
    "authored_surrogate_builder",
    "PROMPT",
]

Config = dict[str, Any]


@dataclass
class AuthorTelemetry:
    """What the authoring replies did. Counted, never inferred."""

    calls: int = 0
    accepted: int = 0
    no_code_block: int = 0
    wrong_entry_point: int = 0
    forbidden_import: int = 0
    unparseable: int = 0
    errors: int = 0
    sources: list = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "accepted": self.accepted,
            "no_code_block": self.no_code_block,
            "wrong_entry_point": self.wrong_entry_point,
            "forbidden_import": self.forbidden_import,
            "unparseable": self.unparseable,
            "errors": self.errors,
            "sources": len(self.sources),
        }


PROMPT = """You are writing the SURROGATE MODEL for a black-box multi-objective \
optimizer. Real evaluations are expensive; your code gives cheap predictions so \
the optimizer can rank candidate configurations before paying to measure them.

OBJECTIVES (name and direction):
{goals}

SEARCH SPACE:
{schema}

Write ONE Python function with EXACTLY this signature:

    {contract}

Rules:
- Use what the parameter NAMES AND MEANINGS tell you about this domain --
  known relationships, ratios, interactions, orderings. That knowledge is the
  only reason your code can beat a generic per-parameter average, which is
  what it will be validated against on held-out data before it is trusted.
- Fit whatever coefficients your structure needs from train_x/train_y inside
  the function. It is called fresh every time with all data measured so far.
- Return one dict per test_x row containing EVERY objective name.
- Standard library only; imports limited to: {imports}.
- No I/O, no globals, deterministic.

Reply with ONLY one fenced Python code block and no other text."""


_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def author_surrogate(
    complete: Callable[[str], str],
    *,
    objectives: Sequence[ObjectiveSpec],
    schema_text: str,
    attempts: int = 2,
    telemetry: Optional[AuthorTelemetry] = None,
) -> Optional[AuthoredArtifact]:
    """Ask the model to write ``fit_predict``; accept whole or not at all."""

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["surrogate"]
    prompt = PROMPT.format(
        goals="\n".join(f"  {s.name}: {s.goal}imise" for s in objectives),
        schema=schema_text,
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
        match = _FENCE.search(text)
        if match is None:
            tel.no_code_block += 1
            continue
        source = match.group(1)
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
        return authored_artifact(
            "surrogate", source, name="llm_surrogate", authored_by="llm"
        )
    return None


REVISION_PROMPT = """You previously wrote this surrogate model for a black-box \
multi-objective optimizer:

```python
{source}
```

It was validated on REAL evaluation data it had never seen, and here is how
it actually performed:

{feedback}

Revise the function. Keep what works, fix what the residuals show is wrong --
a systematic bias on one objective means a missing term or interaction, not a
scaling constant. Same rules as before: exactly this signature

    {contract}

fit whatever coefficients your structure needs from train_x/train_y inside
the function; return one dict per test_x row with EVERY objective name;
standard library only ({imports}); deterministic; no I/O.

Reply with ONLY one fenced Python code block and no other text."""


def render_validation_feedback(
    verdict: Any,
    evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
    predictions: Optional[Sequence[Mapping[str, float]]] = None,
    *,
    worst: int = 5,
) -> str:
    """The residual story a revision needs, as text.

    Per-objective error against the trivial train-mean baseline, plus the
    worst individual mispredictions when the caller has them -- the concrete
    cases a systematic bias shows up in. Pure rendering; never raises.
    """

    lines = []
    mse = getattr(verdict, "per_objective_mse", {}) or {}
    baseline = getattr(verdict, "baseline_mse", {}) or {}
    for name in mse:
        base = baseline.get(name)
        ratio = (f"{mse[name] / base:.2f}x the train-mean baseline"
                 if base else "baseline unavailable")
        lines.append(f"  {name}: holdout MSE {mse[name]:.6g} ({ratio}; "
                     "below 1.00x is predictive)")
    detail = getattr(verdict, "detail", "")
    if detail:
        lines.append(f"  gate detail: {detail}")
    if predictions and evaluated:
        residuals = []
        for (config, actual), predicted in zip(evaluated, predictions):
            if not isinstance(predicted, Mapping):
                continue
            for name, value in actual.items():
                guess = predicted.get(name)
                if isinstance(guess, (int, float)):
                    residuals.append((abs(float(guess) - float(value)),
                                      name, config, guess, value))
        residuals.sort(reverse=True, key=lambda row: row[0])
        for _size, name, config, guess, value in residuals[:worst]:
            lines.append(f"  worst: {name} predicted {guess:.6g}, actual "
                         f"{value:.6g} at {json_compact(config)}")
    return "\n".join(lines) if lines else "  (no usable validation detail)"


def json_compact(config: Config) -> str:
    import json as _json

    text = _json.dumps(config, sort_keys=True, default=str)
    return text if len(text) <= 200 else text[:197] + "..."


def revise_surrogate(
    complete: Callable[[str], str],
    *,
    artifact: AuthoredArtifact,
    feedback: str,
    attempts: int = 1,
    telemetry: Optional[AuthorTelemetry] = None,
) -> Optional[AuthoredArtifact]:
    """One revision round: the artifact, its measured failures, a rewrite.

    The gate treats a revision exactly like a fresh authorship: fenced block
    only, import allowlist, correct entry point, whole-reply rejection. A
    revision that fails yields None and the caller keeps what it had.
    """

    tel = telemetry if telemetry is not None else AuthorTelemetry()
    contract = CONTRACTS["surrogate"]
    prompt = REVISION_PROMPT.format(
        source=artifact.source,
        feedback=feedback,
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
        match = _FENCE.search(text)
        if match is None:
            tel.no_code_block += 1
            continue
        source = match.group(1)
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
        return authored_artifact(
            "surrogate", source,
            name=f"{artifact.name}_rev", authored_by="llm",
        )
    return None


def authored_surrogate_builder(
    artifact: AuthoredArtifact,
    runtime: AuthoredRuntime,
    *,
    on_failure: Optional[Callable[[str], None]] = None,
) -> SurrogateBuilder:
    """Wrap an authored artifact behind the standard builder interface.

    The artifact is stateless by contract (fit + predict in one call), so the
    builder's "fit" is just closing over the training data; every predict
    batch ships train and test together to one out-of-process worker. Any
    non-ok outcome predicts nothing, which the gate and the screen both treat
    as "sit out" -- an authored crash can cost a generation its screening,
    never its budget and never the run.
    """

    def build(
        evaluated: Sequence[Tuple[Config, Mapping[str, float]]],
        specs: Sequence[ObjectiveSpec],
    ) -> Predict:
        train_x = [dict(cfg) for cfg, _obj in evaluated]
        train_y = [dict(obj) for _cfg, obj in evaluated]

        def predict(pool: Sequence[Config]):
            outcome = runtime.call(
                artifact, [[train_x, train_y, [dict(c) for c in pool]]]
            )
            if not outcome.ok:
                if on_failure is not None:
                    on_failure(f"{outcome.status}: {outcome.detail}"[:200])
                return None
            [rows] = outcome.results
            if not isinstance(rows, list):
                return None
            return rows

        return predict

    return build

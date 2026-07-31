"""A proposer that samples the candidate schema instead of calling a model.

This exists for two reasons and serves both equally.

It is the **credential-free path**: someone can install the package and watch
the loop run end to end -- seeds, validation, materialization, evaluation, a
Pareto front -- before deciding whether to spend anything on a provider.

It is also the **honest baseline**. Uninformed sampling over the same schema,
the same budget and the same evaluator is the control every claim in this
project is measured against, and on one benchmark a single median random draw
already accounted for roughly 80 percent of the hypervolume that a full
optimization run produced. Anyone deciding whether a model is worth paying for
on *their* problem should run this arm first. ``agent_evolve check`` does
exactly that.

Sampling is derived from the candidate model's JSON schema, so it needs no
problem-specific code. It is deliberately uninformed: it never reads the Pareto
front, the failures, or the insights, because a baseline that learned would not
be a baseline.
"""

from __future__ import annotations

import random
import string
from typing import Any, Dict, List, Optional, Sequence

from agent_evolve.harness.base import HarnessBase

__all__ = ["RandomProposer"]

_MAX_TRIES_PER_CANDIDATE = 24


class RandomProposer(HarnessBase):
    """Uninformed sampler over the candidate schema. No provider, no network."""

    id = "random"

    #: This proposer synthesises no guidance. Declared rather than
    #: discovered, so the loop skips the insight calls instead of warning
    #: about an empty return that is correct.
    provides_insights = False

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self.calls: List[str] = []

    # -- the seven harness operations -------------------------------------
    # Candidate ops sample; insight ops return nothing, because an uninformed
    # baseline that produced guidance would not be uninformed.

    def generate_initial(self, n: int) -> List[Dict[str, Any]]:
        return self._sample(n, "generate_initial")

    def regenerate(
        self,
        failed_str: str,
        n: int,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[Dict[str, Any]]:
        return self._sample(n, "regenerate")

    def generate_offspring(
        self,
        pareto_str: str,
        n: int,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[Dict[str, Any]]:
        return self._sample(n, "generate_offspring")

    def regenerate_offspring(
        self,
        failed_str: str,
        pareto_str: str,
        n: int,
        constraint_instruction: str,
        performance_insights: str,
    ) -> List[Dict[str, Any]]:
        return self._sample(n, "regenerate_offspring")

    def failure_insights(self, failed_str: str, n_failed: int) -> List[str]:
        return []

    def constraint_instruction(self, failed_str: str, previous: Optional[str] = None) -> str:
        return previous or ""

    def performance_insights(
        self,
        stats_str: str,
        pareto_str: str,
        previous: Optional[str] = None,
    ) -> str:
        return previous or ""

    # -- sampling ----------------------------------------------------------

    def _sample(self, n: int, op: str) -> List[Dict[str, Any]]:
        self.calls.append(op)
        schema = self._schema()
        return [self._draw_object(schema, schema) for _ in range(max(0, n))]

    def _schema(self) -> Dict[str, Any]:
        ctx = self._ctx
        model = getattr(ctx, "candidate_model", None) if ctx is not None else None
        if model is None:
            return {}
        try:
            return model.model_json_schema()
        except Exception:
            return {}

    def _resolve(self, node: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
        seen = 0
        while isinstance(node, dict) and "$ref" in node and seen < 8:
            ref = node["$ref"]
            seen += 1
            if not ref.startswith("#/"):
                return {}
            target: Any = root
            for part in ref[2:].split("/"):
                if not isinstance(target, dict) or part not in target:
                    return {}
                target = target[part]
            node = target if isinstance(target, dict) else {}
        return node if isinstance(node, dict) else {}

    def _draw_object(self, node: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
        node = self._resolve(node, root)
        props = node.get("properties") or {}
        required = set(node.get("required") or props.keys())
        out: Dict[str, Any] = {}
        for name, sub in props.items():
            if name in required:
                out[name] = self._draw(sub, root)
        return out

    def _draw(self, node: Dict[str, Any], root: Dict[str, Any], depth: int = 0) -> Any:
        node = self._resolve(node, root)
        if depth > 6:
            return None
        if "enum" in node and node["enum"]:
            return self._rng.choice(list(node["enum"]))
        if "const" in node:
            return node["const"]
        for key in ("anyOf", "oneOf"):
            if node.get(key):
                return self._draw(self._rng.choice(list(node[key])), root, depth + 1)

        kind = node.get("type")
        if isinstance(kind, list):
            kind = self._rng.choice([k for k in kind if k != "null"] or ["string"])

        if kind == "integer":
            lo = node.get("minimum", node.get("exclusiveMinimum", 0))
            hi = node.get("maximum", node.get("exclusiveMaximum", lo + 16))
            lo, hi = int(lo), int(hi)
            return self._rng.randint(lo, hi) if hi >= lo else lo
        if kind == "number":
            lo = float(node.get("minimum", node.get("exclusiveMinimum", 0.0)))
            hi = float(node.get("maximum", node.get("exclusiveMaximum", lo + 1.0)))
            return self._rng.uniform(lo, hi) if hi >= lo else lo
        if kind == "boolean":
            return self._rng.random() < 0.5
        if kind == "array":
            items = node.get("items") or {}
            lo = int(node.get("minItems", 1))
            hi = int(node.get("maxItems", max(lo, 6)))
            count = self._rng.randint(lo, hi) if hi >= lo else lo
            drawn = [self._draw(items, root, depth + 1) for _ in range(count)]
            if node.get("uniqueItems"):
                unique: List[Any] = []
                for value in drawn:
                    if value not in unique:
                        unique.append(value)
                drawn = unique
            return drawn
        if kind == "object":
            return self._draw_object(node, root)
        if kind == "null":
            return None

        alphabet = string.ascii_lowercase
        return "".join(self._rng.choice(alphabet) for _ in range(self._rng.randint(3, 8)))

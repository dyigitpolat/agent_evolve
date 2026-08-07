"""The fossil: the credential-free genetic path's exact evaluate-call stream.

Pinned from the tree as of 2026-08-07, BEFORE the authorship substrate landed.
Every mechanism added since (weighted priors, surrogate screening, operator
portfolios, authored code) is off by default, and "off" is defined here as
byte-identical: same candidates, in the same order, through the public entry
point, for two seeds. A refactor that reorders RNG draws or slips an extra
sample into the stream fails this file even when every quality metric looks
fine — which is the point: quality metrics cannot see a changed control arm.

If this test fails, the default path's behaviour changed. That is either a
deliberate, announced break (re-pin the literals in the same commit and say so
in the commit message) or a bug. There is no third reading.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Literal, Mapping, Sequence

from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome


class _FossilCandidate(BaseModel):
    genome: list[Literal[0, 1]]


class _FossilProblem:
    candidate_model = _FossilCandidate
    objectives = (
        ObjectiveSpec(name="ones", goal="max"),
        ObjectiveSpec(name="edges", goal="min"),
    )

    def __init__(self) -> None:
        self.stream: list[list[int]] = []

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0, 0, 0, 0, 0, 0, 0, 0]},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.stream.append(list(artifact))
        ones = float(sum(artifact))
        edges = float(
            sum(1 for i in range(len(artifact) - 1) if artifact[i] != artifact[i + 1])
        )
        return {"ones": ones, "edges": edges}


#: (seed) -> (sha256 of the evaluate-call stream, stream length, evaluations).
#: Computed on the pre-substrate tree; see the module docstring for the rules.
_FOSSILS = {
    11: ("79bab67d6c28753b37c8fc7ca58dc25a85cd5cd6a378d6ff32d8f487e6d918f1", 16, 16),
    23: ("69c969bd121326723047a010090a6a7e0ef37926320d91ca1e969244b22d7133", 16, 16),
}


def test_default_run_reproduces_the_pre_substrate_stream() -> None:
    for seed, (digest, length, evaluations) in _FOSSILS.items():
        problem = _FossilProblem()
        result = optimize(problem, budget=16, seed=seed)
        actual = hashlib.sha256(
            json.dumps(problem.stream, separators=(",", ":")).encode()
        ).hexdigest()
        assert (actual, len(problem.stream), result.evaluations) == (
            digest, length, evaluations,
        ), (
            f"seed {seed}: the credential-free default stream changed "
            f"(got {actual[:12]}…, {len(problem.stream)} calls, "
            f"{result.evaluations} charged). Either announce the break and "
            "re-pin, or find the mechanism that is not off by default."
        )

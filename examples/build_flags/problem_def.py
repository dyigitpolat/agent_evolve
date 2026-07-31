"""Compiler flag selection: a problem in the regime where this tool is strongest.

This is the example to read if you want to know whether ``agent_evolve`` suits
your problem, because it sits where the measured advantage is largest:

* **categorical**, not continuous -- flags are choices, not dials;
* **constrained** in ways a sampler cannot guess -- some combinations are
  contradictory, and one is rejected outright;
* **semantically structured** -- the flag names mean something, which is the
  only thing a language model brings that a random sampler cannot;
* **expensive to evaluate** -- a real build-and-benchmark cycle takes minutes,
  which is the regime where per-call model overhead is repaid.

The evaluator here is a deterministic *stand-in*, not a real compiler, so the
example runs anywhere in milliseconds with no toolchain. Swap ``evaluate`` for
a real build and benchmark and nothing else changes -- that substitution is the
entire integration cost.

Run it free::

    python examples/build_flags/run.py
    agent_evolve check examples.build_flags.problem_def:problem --baseline-only
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from agent_evolve import ObjectiveSpec, ValidationOutcome

OptLevel = Literal["O0", "O1", "O2", "O3", "Os", "Ofast"]
Vectorize = Literal["none", "sse4", "avx2", "avx512"]
LinkTime = Literal["none", "thin", "full"]


class BuildConfig(BaseModel):
    """One candidate build configuration."""

    opt_level: OptLevel = Field(..., description="optimization level")
    vectorize: Vectorize = Field(..., description="vector instruction set")
    link_time: LinkTime = Field(..., description="link-time optimization mode")
    unroll_loops: bool = Field(..., description="enable aggressive loop unrolling")
    fast_math: bool = Field(..., description="permit unsafe float reassociation")
    inline_threshold: int = Field(..., ge=0, le=1000, description="inliner cost threshold")


class BuildFlags:
    """Choose compiler flags: fast binary, small binary, correct results."""

    candidate_model = BuildConfig

    @property
    def objectives(self):
        return [
            ObjectiveSpec("runtime_ms", "min"),
            ObjectiveSpec("binary_kb", "min"),
        ]

    def seeds(self):
        """What a person would actually try first: the two stock recipes."""
        return [
            {
                "opt_level": "O2", "vectorize": "none", "link_time": "none",
                "unroll_loops": False, "fast_math": False, "inline_threshold": 225,
            },
            {
                "opt_level": "Os", "vectorize": "none", "link_time": "thin",
                "unroll_loops": False, "fast_math": False, "inline_threshold": 75,
            },
        ]

    def validate(self, config) -> ValidationOutcome:
        """The constraints a sampler cannot guess but a reader can.

        Each message names the offending combination and what would be
        acceptable, because the message is fed back to the proposer verbatim.
        """
        opt = config.get("opt_level")
        if opt == "O0" and config.get("unroll_loops"):
            return ValidationOutcome(
                False,
                "constraint",
                "unroll_loops has no effect at O0 and the build rejects it; "
                "either raise opt_level to O2 or above, or disable unroll_loops",
            )
        if opt == "Ofast" and not config.get("fast_math"):
            return ValidationOutcome(
                False,
                "constraint",
                "Ofast implies fast_math; set fast_math true or choose O3 instead",
            )
        if opt == "Os" and config.get("inline_threshold", 0) > 200:
            return ValidationOutcome(
                False,
                "constraint",
                "Os targets size, so an inline_threshold above 200 contradicts it; "
                "use at most 200, or switch to O2/O3",
            )
        if config.get("vectorize") == "avx512" and opt in ("O0", "O1"):
            return ValidationOutcome(
                False,
                "constraint",
                "avx512 needs O2 or above to be emitted at all; raise opt_level "
                "or drop to avx2",
            )
        return ValidationOutcome(True)

    def materialize(self, config) -> tuple:
        """Canonicalise to the flags that actually reach the compiler.

        ``inline_threshold`` is quantised to the 25-unit steps the inliner
        genuinely distinguishes, and it is ignored entirely at O0. Two
        configurations differing only below that resolution produce the *same
        build*, so they are evaluated once rather than twice. On a real
        toolchain that is minutes of build time, not microseconds.
        """
        opt = config["opt_level"]
        threshold = 0 if opt == "O0" else round(config["inline_threshold"] / 25) * 25
        return (
            opt,
            config["vectorize"],
            config["link_time"],
            bool(config["unroll_loops"]),
            bool(config["fast_math"]),
            threshold,
        )

    def evaluate(self, artifact) -> dict:
        """Stand-in for a build and benchmark. Deterministic, no toolchain.

        Replace this with a real build and timing run; nothing else changes.
        """
        opt, vec, lto, unroll, fast, threshold = artifact

        runtime = 1000.0
        runtime *= {"O0": 2.4, "O1": 1.35, "O2": 1.0, "O3": 0.93, "Os": 1.12, "Ofast": 0.88}[opt]
        runtime *= {"none": 1.0, "sse4": 0.94, "avx2": 0.86, "avx512": 0.83}[vec]
        runtime *= {"none": 1.0, "thin": 0.97, "full": 0.94}[lto]
        runtime *= 0.96 if unroll else 1.0
        runtime *= 0.95 if fast else 1.0
        runtime *= 1.0 - min(threshold, 600) / 12000.0

        size = 300.0
        size *= {"O0": 1.0, "O1": 1.08, "O2": 1.22, "O3": 1.5, "Os": 0.82, "Ofast": 1.55}[opt]
        size *= {"none": 1.0, "sse4": 1.04, "avx2": 1.1, "avx512": 1.18}[vec]
        size *= {"none": 1.0, "thin": 0.93, "full": 0.86}[lto]
        size *= 1.14 if unroll else 1.0
        size *= 1.0 + min(threshold, 1000) / 2200.0

        # A little deterministic roughness, so the landscape is not a smooth
        # bowl that any hill-climber trivially solves.
        jitter = int(hashlib.sha256(repr(artifact).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        runtime *= 0.97 + 0.06 * jitter
        size *= 0.98 + 0.04 * jitter

        return {"runtime_ms": round(runtime, 2), "binary_kb": round(size, 2)}

    def search_space_description(self) -> str:
        return (
            "Choose compiler flags for a numeric kernel, trading execution time "
            "against binary size.\n\n"
            "opt_level: O0, O1, O2, O3, Os (size), Ofast (implies fast math)\n"
            "vectorize: none, sse4, avx2, avx512\n"
            "link_time: none, thin, full\n"
            "unroll_loops: bool -- faster, larger; no effect at O0\n"
            "fast_math: bool -- faster, relaxes float semantics; required by Ofast\n"
            "inline_threshold: 0-1000 -- higher inlines more; larger binary\n\n"
            "Minimise runtime_ms and binary_kb together; they trade off, so a "
            "front of several configurations is the useful answer."
        )


problem = BuildFlags()

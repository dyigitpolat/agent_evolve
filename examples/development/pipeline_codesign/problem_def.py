"""Development-only structured compiler/runtime co-design landscape.

This problem is intentionally cheap and deterministic.  It is used to debug
variation, lineage, memory, and trace behavior; it is never a paper benchmark or
evidence for wall-clock dominance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_evolve import ObjectiveSpec


PassName = Literal["simplify", "inline", "licm", "unroll", "vectorize", "dce"]


class FrontendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inline_threshold: Literal[0, 20, 40, 60, 80]
    unroll_factor: Literal[1, 2, 4, 8]
    vector_width: Literal[1, 2, 4, 8]


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    register_budget: Literal[16, 24, 32, 40, 48, 64]
    scheduler: Literal["compact", "balanced", "aggressive"]


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threads: Literal[1, 2, 4, 8]
    prefetch_distance: Literal[0, 2, 4, 8, 16]
    data_layout: Literal["aos", "soa", "blocked"]


class CandidateConfig(BaseModel):
    """One typed compiler/runtime configuration."""

    model_config = ConfigDict(extra="forbid")

    passes: list[PassName] = Field(min_length=2, max_length=6)
    frontend: FrontendConfig
    backend: BackendConfig
    runtime: RuntimeConfig


BASE_CONFIG = {
    "passes": ["simplify", "dce"],
    "frontend": {
        "inline_threshold": 20,
        "unroll_factor": 1,
        "vector_width": 1,
    },
    "backend": {"register_budget": 24, "scheduler": "balanced"},
    "runtime": {"threads": 1, "prefetch_distance": 0, "data_layout": "aos"},
}

# Frozen branch cohort for paired operator debugging.  LEFT changes compiler
# components, RIGHT changes only runtime, and their exact union is a valid child
# that dominates both on at least one objective without worsening the inherited
# cost component.  These are test fixtures, not benchmark incumbents.
DEVELOPMENT_BRANCH_LEFT = {
    "passes": ["simplify", "licm", "unroll", "vectorize", "dce"],
    "frontend": {
        "inline_threshold": 20,
        "unroll_factor": 2,
        "vector_width": 4,
    },
    "backend": {"register_budget": 40, "scheduler": "aggressive"},
    "runtime": dict(BASE_CONFIG["runtime"]),
}
DEVELOPMENT_BRANCH_RIGHT = {
    "passes": list(BASE_CONFIG["passes"]),
    "frontend": dict(BASE_CONFIG["frontend"]),
    "backend": dict(BASE_CONFIG["backend"]),
    "runtime": {"threads": 4, "prefetch_distance": 4, "data_layout": "blocked"},
}
DEVELOPMENT_RECOMBINATION_TARGET = {
    "passes": list(DEVELOPMENT_BRANCH_LEFT["passes"]),
    "frontend": dict(DEVELOPMENT_BRANCH_LEFT["frontend"]),
    "backend": dict(DEVELOPMENT_BRANCH_LEFT["backend"]),
    "runtime": dict(DEVELOPMENT_BRANCH_RIGHT["runtime"]),
}

# Exhaustive development oracle with every non-runtime field fixed to the exact
# recombination target above.  This constant is a workflow kill-test target,
# not benchmark evidence: the synthetic evaluator is fully disclosed to the
# development harness and the 4 * 5 * 3 runtime grid is only 60 points.
DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM = {
    "passes": list(DEVELOPMENT_RECOMBINATION_TARGET["passes"]),
    "frontend": dict(DEVELOPMENT_RECOMBINATION_TARGET["frontend"]),
    "backend": dict(DEVELOPMENT_RECOMBINATION_TARGET["backend"]),
    "runtime": {
        "threads": 8,
        "prefetch_distance": 4,
        "data_layout": "soa",
    },
}
DEVELOPMENT_RUNTIME_SCOPED_OPTIMUM_OBJECTIVES = {
    "speedup": 3.86,
    "code_size": 116.4,
    "compile_time": 17.49,
}


class PipelineCoDesignProblem:
    """Nonlinear structured landscape with useful cross-component recombination."""

    candidate_model = CandidateConfig
    example_config = BASE_CONFIG
    constraints_description = (
        "Passes must be unique. LICM must precede vectorize; vectorize requires "
        "vector_width > 1. The unroll pass requires unroll_factor > 1. The "
        "register-pressure bound must be respected."
    )

    @property
    def objectives(self):
        return [
            ObjectiveSpec("speedup", "max"),
            ObjectiveSpec("code_size", "min"),
            ObjectiveSpec("compile_time", "min"),
        ]

    @staticmethod
    def _pressure(config) -> int:
        frontend = config["frontend"]
        runtime = config["runtime"]
        return (
            8
            + 2 * frontend["unroll_factor"]
            + 2 * frontend["vector_width"]
            + frontend["inline_threshold"] // 10
            + runtime["threads"]
        )

    def validate(self, config):
        # Re-validate mappings supplied outside the Pydantic generator boundary.
        parsed = CandidateConfig.model_validate(config)
        normalized = parsed.model_dump(mode="python")
        passes = normalized["passes"]
        if len(passes) != len(set(passes)):
            raise ValueError("passes must not contain duplicates")
        if "vectorize" in passes:
            if "licm" not in passes or passes.index("licm") > passes.index("vectorize"):
                raise ValueError("vectorize requires LICM earlier in the pass order")
            if normalized["frontend"]["vector_width"] == 1:
                raise ValueError("vectorize requires vector_width > 1")
        if "unroll" in passes and normalized["frontend"]["unroll_factor"] == 1:
            raise ValueError("the unroll pass requires unroll_factor > 1")
        if self._pressure(normalized) > normalized["backend"]["register_budget"]:
            raise ValueError(
                "register pressure exceeds backend.register_budget; reduce inline, "
                "unroll, vector width, or threads, or raise the budget"
            )
        return True

    def evaluate(self, config):
        parsed = CandidateConfig.model_validate(config).model_dump(mode="python")
        self.validate(parsed)
        passes = parsed["passes"]
        frontend = parsed["frontend"]
        backend = parsed["backend"]
        runtime = parsed["runtime"]

        pass_gain = {
            "simplify": 0.08,
            "inline": 0.12,
            "licm": 0.18,
            "unroll": 0.16,
            "vectorize": 0.28,
            "dce": 0.05,
        }
        speedup = 1.0 + sum(pass_gain[name] for name in passes)
        speedup += 0.0025 * frontend["inline_threshold"]
        speedup += 0.055 * (frontend["unroll_factor"] - 1)
        speedup += 0.075 * (frontend["vector_width"] - 1)
        speedup += 0.11 * (runtime["threads"] - 1)

        # Cross-component innovations are deliberately valuable: these are what
        # lineage-aware recombination should preserve and combine.
        if "licm" in passes and "unroll" in passes:
            speedup += 0.24
        if "vectorize" in passes and runtime["data_layout"] == "soa":
            speedup += 0.38
        if runtime["threads"] >= 4 and runtime["data_layout"] == "blocked":
            speedup += 0.31
        if runtime["prefetch_distance"] == 4 and runtime["threads"] >= 2:
            speedup += 0.21
        if backend["scheduler"] == "aggressive" and "licm" in passes:
            speedup += 0.18
        if backend["scheduler"] == "compact" and frontend["inline_threshold"] >= 60:
            speedup -= 0.25

        pressure = self._pressure(parsed)
        slack = backend["register_budget"] - pressure
        if slack < 4:
            speedup -= 0.17
        elif slack > 24:
            speedup -= 0.06

        pass_size = {
            "simplify": -1.0,
            "inline": 8.0,
            "licm": 2.0,
            "unroll": 6.0,
            "vectorize": 5.0,
            "dce": -3.0,
        }
        code_size = (
            100.0
            + sum(pass_size[name] for name in passes)
            + 0.16 * frontend["inline_threshold"]
            + 1.8 * (frontend["unroll_factor"] - 1)
            + 0.8 * (frontend["vector_width"] - 1)
        )
        if "inline" in passes and "dce" in passes and passes.index("inline") < passes.index("dce"):
            code_size -= 5.0

        scheduler_cost = {"compact": 0.5, "balanced": 1.5, "aggressive": 3.0}
        compile_time = (
            5.0
            + 1.35 * len(passes)
            + 0.035 * frontend["inline_threshold"]
            + 0.42 * frontend["unroll_factor"]
            + 0.3 * frontend["vector_width"]
            + scheduler_cost[backend["scheduler"]]
        )
        return {
            "speedup": round(float(speedup), 6),
            "code_size": round(float(code_size), 6),
            "compile_time": round(float(compile_time), 6),
        }

    def search_space_description(self):
        return """Development-only compiler/runtime co-design.

Choose an ordered unique subset (length 2..6) of passes:
  simplify, inline, licm, unroll, vectorize, dce

Tune three nested components:
  frontend.inline_threshold in {0,20,40,60,80}
  frontend.unroll_factor in {1,2,4,8}
  frontend.vector_width in {1,2,4,8}
  backend.register_budget in {16,24,32,40,48,64}
  backend.scheduler in {compact,balanced,aggressive}
  runtime.threads in {1,2,4,8}
  runtime.prefetch_distance in {0,2,4,8,16}
  runtime.data_layout in {aos,soa,blocked}

Maximize speedup while minimizing code_size and compile_time. Useful interactions
exist between pass order, frontend transformations, backend scheduling, data
layout, prefetching, and thread count. Extreme settings can lose performance or
violate register pressure; a strong candidate balances all three objectives."""

    @staticmethod
    def render_candidate(config):
        return CandidateConfig.model_validate(config).model_dump_json()


problem = PipelineCoDesignProblem()

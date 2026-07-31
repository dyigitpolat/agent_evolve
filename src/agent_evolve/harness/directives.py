"""The Directives port: problem-supplied prompt text.

A ``Directives`` provider supplies *what to say* to the LLM while harness
adapters supply *how to run it*. The backbone ships
:class:`DefaultDirectives` (generic wording) so examples and new problems work
out of the box; a problem can expose its own ``directives`` attribute to fully
own its prompts.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

#: Chain-of-thought instruction, paired with a Proposal(thought_process,
#: candidates) output type so the model reasons before committing.
#:
#: This wording used to instruct the model to "split its factors across the
#: levels so every per-dimension product is EXACTLY correct" -- the constraint
#: of the one problem the backbone was first written against, shipped as if it
#: were generic. Every other problem received an instruction about a structure
#: it does not have. ``DefaultDirectives`` is the *problem-agnostic* wording by
#: definition; anything specific to one search space belongs in that problem's
#: own ``Directives``, which is exactly what the port is for.
_COT = (
    " First, in thought_process, reason step by step as a mental draft: state what the "
    "measurements so far imply about which parts of the configuration matter, decide what "
    "to change and by how much, and check each proposal against the stated constraints and "
    "the declared schema. Then fill candidates."
)


@runtime_checkable
class Directives(Protocol):
    """Prompt wording for each operation."""

    def compose_initial(self, context: str, n_candidates: int) -> str: ...
    def compose_regenerate(
        self, context: str, failed_str: str, n_candidates: int,
        constraint_instruction: str, performance_insights: str,
    ) -> str: ...
    def compose_offspring(
        self, context: str, pareto_str: str, n_candidates: int,
        constraint_instruction: str, performance_insights: str,
    ) -> str: ...
    def compose_regenerate_offspring(
        self, context: str, failed_str: str, pareto_str: str, n_candidates: int,
        constraint_instruction: str, performance_insights: str,
    ) -> str: ...
    def compose_failure_insights(self, context: str, failed_str: str, n_failed: int) -> str: ...
    def compose_constraint_instruction(self, context: str, failed_str: str, previous: str = "") -> str: ...
    def compose_performance_insights(
        self, context: str, stats_str: str, pareto_str: str, previous: str = ""
    ) -> str: ...


class DefaultDirectives:
    """Generic, problem-agnostic prompt wording.

    Problem specifics (search space, constraints, examples) reach the prompt
    through ``context`` (built from the problem's hooks), so this generic wording
    works for any problem; a problem may still supply its own ``Directives``.
    """

    def compose_initial(self, context: str, n_candidates: int) -> str:
        return (
            "You are an optimization expert generating candidates for a multi-objective problem.\n\n"
            f"{context}\n\n"
            f"Propose exactly {n_candidates} diverse, valid candidates that explore different "
            "regions of the search space and trade off the objectives."
            + _COT
        )

    def compose_regenerate(
        self, context: str, failed_str: str, n_candidates: int,
        constraint_instruction: str, performance_insights: str,
    ) -> str:
        return (
            "You are an optimization expert. Previous candidates failed validation; learn from "
            "them and propose better ones.\n\n"
            f"{context}\n\n"
            f"FAILED CANDIDATES AND THEIR ISSUES:\n{failed_str}\n\n"
            f"CONSTRAINT COMPLIANCE INSTRUCTIONS:\n{constraint_instruction or 'None yet.'}\n\n"
            f"PERFORMANCE INSIGHTS:\n{performance_insights or 'None yet.'}\n\n"
            f"Propose exactly {n_candidates} NEW valid candidates that fix the failures and follow "
            "the constraint instructions."
            + _COT
        )

    def compose_offspring(
        self, context: str, pareto_str: str, n_candidates: int,
        constraint_instruction: str, performance_insights: str,
    ) -> str:
        return (
            "You are an optimization expert. Breed offspring from high-quality Pareto-optimal "
            "configurations.\n\n"
            f"{context}\n\n"
            f"PARETO-OPTIMAL CONFIGURATIONS (best so far):\n{pareto_str}\n\n"
            f"CONSTRAINT COMPLIANCE INSTRUCTIONS:\n{constraint_instruction or 'Follow standard constraints.'}\n\n"
            f"PERFORMANCE INSIGHTS:\n{performance_insights or 'Analyze the Pareto set for patterns.'}\n\n"
            f"Propose exactly {n_candidates} NEW valid candidates that build on the Pareto patterns, "
            "explore new trade-offs, and try to improve on existing solutions."
            + _COT
        )

    def compose_regenerate_offspring(
        self, context: str, failed_str: str, pareto_str: str, n_candidates: int,
        constraint_instruction: str, performance_insights: str,
    ) -> str:
        return (
            "You are an optimization expert. Some offspring failed; fix them while staying close to "
            "the known-valid Pareto set.\n\n"
            f"{context}\n\n"
            f"PARETO-OPTIMAL CONFIGURATIONS (known valid, high quality):\n{pareto_str}\n\n"
            f"FAILED OFFSPRING AND THEIR ISSUES:\n{failed_str}\n\n"
            f"CONSTRAINT COMPLIANCE INSTRUCTIONS:\n{constraint_instruction or 'Follow the Pareto patterns.'}\n\n"
            f"PERFORMANCE INSIGHTS:\n{performance_insights or ''}\n\n"
            f"Propose exactly {n_candidates} NEW valid candidates that fix the failures and try to improve."
            + _COT
        )

    def compose_failure_insights(self, context: str, failed_str: str, n_failed: int) -> str:
        return (
            "You are an optimization expert. Analyze why each candidate failed and give one "
            "specific, actionable fix per candidate.\n\n"
            f"{context}\n\n"
            f"FAILED CANDIDATES:\n{failed_str}\n\n"
            f"Provide exactly {n_failed} insight strings, one per failed candidate, in order."
        )

    def compose_constraint_instruction(self, context: str, failed_str: str, previous: str = "") -> str:
        if previous:
            return (
                "You are an optimization expert. Update the constraint-compliance guide using new failures.\n\n"
                f"PREVIOUS GUIDE:\n{previous}\n\n"
                f"NEW FAILED CANDIDATES AND INSIGHTS:\n{failed_str}\n\n"
                "Produce an updated, comprehensive constraint-compliance guide."
            )
        return (
            "You are an optimization expert. From these failures, write a consolidated "
            "constraint-compliance guide future candidates should follow.\n\n"
            f"{context}\n\n"
            f"FAILED CANDIDATES AND INSIGHTS:\n{failed_str}\n\n"
            "Produce a clear, actionable constraint-compliance guide."
        )

    def compose_performance_insights(
        self, context: str, stats_str: str, pareto_str: str, previous: str = ""
    ) -> str:
        if previous:
            return (
                "You are an optimization expert. Update the performance insights with new results.\n\n"
                f"PREVIOUS INSIGHTS:\n{previous}\n\n"
                f"CURRENT PERFORMANCE STATISTICS:\n{stats_str}\n\n"
                f"CURRENT TOP PARETO CONFIGURATIONS:\n{pareto_str}\n\n"
                "Produce updated performance insights that capture new patterns."
            )
        return (
            "You are an optimization expert. Analyze the performance patterns to guide better candidates.\n\n"
            f"{context}\n\n"
            f"PERFORMANCE STATISTICS:\n{stats_str}\n\n"
            f"TOP PARETO CONFIGURATIONS:\n{pareto_str}\n\n"
            "Explain what makes configurations perform well, the trade-offs, and how to improve."
        )

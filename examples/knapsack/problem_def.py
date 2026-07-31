"""Multi-objective knapsack: the five obligations, worked end to end.

This is the reference integration. It implements exactly the five things
``agent_evolve`` asks for and nothing else:

    candidate_model   the schema a proposal must satisfy
    objectives        what is optimized, and which way
    seeds()           where to start
    validate()        cheap rejection that explains itself
    materialize()     candidate -> the artifact that gets measured
    evaluate()        artifact -> objective values

Run it with no credentials and no cost::

    python examples/knapsack/run.py --proposer random

Knapsack is a *bad* showcase for this optimizer and is used here anyway,
because it is the shortest problem that exercises all five obligations. It is
cheap to evaluate, which is the regime where the LLM's per-call overhead is
never repaid. See ``docs/scope.md`` before concluding anything from it.
"""

from typing import Annotated

from pydantic import BaseModel, Field
from pydantic import Field as Bound

from agent_evolve import ObjectiveSpec, ValidationOutcome


class CandidateConfig(BaseModel):
    """One candidate configuration.

    The bounds are declared, not merely described in prose. Anything that reads
    the schema -- a model, or the uninformed sampler the baseline uses -- then
    draws only legal indices. Leaving them open would hand the model an
    advantage the baseline never had, and comparing those two is the whole
    point of ``agent_evolve check``.
    """

    selection: list[Annotated[int, Bound(ge=0, le=9)]] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Distinct item indices, 0-9, each used at most once",
        json_schema_extra={"uniqueItems": True},
    )


class Knapsack:
    """0/1 knapsack: maximise value, minimise weight, under a capacity bound."""

    candidate_model = CandidateConfig

    ITEMS = [
        # (weight, value)
        (10, 60), (20, 100), (30, 120), (15, 75), (25, 90),
        (5, 40), (35, 150), (12, 55), (18, 80), (8, 45),
    ]
    CAPACITY = 60

    # -- 1. what is being optimized ---------------------------------------
    @property
    def objectives(self):
        return [
            ObjectiveSpec("total_value", "max"),
            ObjectiveSpec("total_weight", "min"),
        ]

    # -- 2. where to start -------------------------------------------------
    def seeds(self):
        """Two configurations a person would try first.

        Seeds are evaluated before anything is proposed, so the result answers
        "did this beat what I already had" rather than leaving it to be
        assumed.
        """
        return [{"selection": [5]}, {"selection": [0, 5, 9]}]

    # -- 3. cheap rejection that explains itself ---------------------------
    def validate(self, config) -> ValidationOutcome:
        selection = config.get("selection")
        if not isinstance(selection, list) or not selection:
            return ValidationOutcome(
                False, "structural", "'selection' must be a non-empty list of item indices"
            )
        for i in selection:
            if not isinstance(i, int) or not 0 <= i < len(self.ITEMS):
                return ValidationOutcome(
                    False,
                    "structural",
                    f"invalid item index {i!r}; use integers 0-{len(self.ITEMS) - 1}",
                )
        if len(selection) != len(set(selection)):
            return ValidationOutcome(
                False, "structural", "each item may be selected at most once"
            )
        weight = sum(self.ITEMS[i][0] for i in selection)
        if weight > self.CAPACITY:
            return ValidationOutcome(
                False,
                "constraint",
                f"total weight {weight} exceeds capacity {self.CAPACITY}; "
                f"drop about {weight - self.CAPACITY} units of weight",
            )
        return ValidationOutcome(True)

    # -- 4. candidate -> the artifact that gets measured -------------------
    def materialize(self, config):
        """Canonicalise to the sorted index tuple.

        ``[5, 1]`` and ``[1, 5]`` are the same knapsack. Materializing both to
        the same artifact means the second one is never paid for. That saving
        is the reason this obligation exists, and on a real problem -- where
        evaluation is a build, a simulation or a benchmark run -- it is the
        difference between spending the budget once and spending it twice.
        """
        return tuple(sorted(config["selection"]))

    # -- 5. measure it -----------------------------------------------------
    def evaluate(self, artifact):
        return {
            "total_value": float(sum(self.ITEMS[i][1] for i in artifact)),
            "total_weight": float(sum(self.ITEMS[i][0] for i in artifact)),
        }

    # -- optional: prose context for a model-driven proposer ---------------
    def search_space_description(self):
        items = "\n".join(
            f"  Item {i}: weight={w}, value={v}"
            for i, (w, v) in enumerate(self.ITEMS)
        )
        return (
            "0/1 knapsack. Choose a subset of item indices; each item may be "
            "used at most once.\n"
            f'Configuration format: {{"selection": [0, 3, 5]}}\n\n'
            f"Available items:\n{items}\n\n"
            f"Total weight must not exceed {self.CAPACITY}. Maximise total "
            "value while minimising total weight."
        )


# The example runner looks for these two names.
problem = Knapsack()
KnapsackProblem = Knapsack  # backwards-compatible alias

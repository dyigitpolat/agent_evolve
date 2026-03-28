"""Multi-objective knapsack problem definition.

This file satisfies the agent_evolve problem_def contract:
  - ``problem`` — an object implementing the Problem protocol
  - ``config``  — search hyper-parameters dict
"""

from agent_evolve import ObjectiveSpec


class KnapsackProblem:
    """0/1 knapsack with two objectives: maximise value, minimise weight."""

    ITEMS = [
        # (weight, value)
        (10, 60),
        (20, 100),
        (30, 120),
        (15, 75),
        (25, 90),
        (5, 40),
        (35, 150),
        (12, 55),
        (18, 80),
        (8, 45),
    ]
    CAPACITY = 60

    @property
    def objectives(self):
        return [
            ObjectiveSpec("total_value", "max"),
            ObjectiveSpec("total_weight", "min"),
        ]

    def validate(self, config):
        selection = config.get("selection", [])
        if not isinstance(selection, list):
            raise ValueError("'selection' must be a list of item indices")
        if not selection:
            raise ValueError("selection must not be empty")
        for i in selection:
            if not isinstance(i, int) or i < 0 or i >= len(self.ITEMS):
                raise ValueError(
                    f"Invalid item index {i}. Must be int in 0..{len(self.ITEMS) - 1}"
                )
        if len(selection) != len(set(selection)):
            raise ValueError("Duplicate item indices are not allowed")
        weight = sum(self.ITEMS[i][0] for i in selection)
        if weight > self.CAPACITY:
            raise ValueError(
                f"Total weight {weight} exceeds capacity {self.CAPACITY}"
            )
        return True

    def evaluate(self, config):
        selection = config["selection"]
        return {
            "total_value": float(sum(self.ITEMS[i][1] for i in selection)),
            "total_weight": float(sum(self.ITEMS[i][0] for i in selection)),
        }

    def search_space_description(self):
        item_desc = "\n".join(
            f"  Item {i}: weight={w}, value={v}"
            for i, (w, v) in enumerate(self.ITEMS)
        )
        return (
            f"0/1 Knapsack Problem\n"
            f"Select a subset of items by index. Each item can be chosen at most once.\n"
            f"Configuration format: {{\"selection\": [0, 3, 5]}}\n\n"
            f"Available items:\n{item_desc}\n\n"
            f"Capacity constraint: total weight must not exceed {self.CAPACITY}.\n"
            f"Objective: maximise total value while minimising total weight."
        )


problem = KnapsackProblem()

config = {
    "pop_size": 6,
    "generations": 3,
    "candidates_per_batch": 4,
    "max_regen_rounds": 5,
    "max_failed_examples": 3,
    "verbose": True,
}

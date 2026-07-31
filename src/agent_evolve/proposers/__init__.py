"""Proposers: where candidates come from.

``random`` needs no credentials and is also the control arm every claim about
a model should be measured against. ``llm`` is the model-driven proposer and
resolves through the harness registry.
"""

from agent_evolve.proposers.random_proposer import RandomProposer

__all__ = ["RandomProposer"]

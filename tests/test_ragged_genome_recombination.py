"""A genome length is a property of a candidate, not of a problem.

`loci_of` gives a sequence-valued field one locus per element, so two candidates
of the same problem can have different genome lengths. The genetic loop used to
read the locus count once, from `seeds[0]`, and reuse it for every parent. On the
shipped knapsack example -- whose seeds are a 1-item and a 3-item selection --
that made a 1-bit crossover mask meet a 3-locus parent, and

    ValueError: mask has 1 bits but the candidate has 3 loci

fired deterministically on the first generation, for every seed, from both the
CLI (`agent_evolve run examples.knapsack.problem_def:problem`) and the example's
own runner (`python examples/knapsack/run.py`). The reference integration in the
README did not run.

Two things are pinned here, because the crash needed both to be wrong:

* `crossover` must be defined on ragged parents -- a locus the donor does not
  have cannot be inherited from it, so the child keeps parent A's value;
* the loop must fit the mask to the parent it is applying it to.

And one thing is pinned that is *not* about the crash: the fix must not have
changed recombination on fixed-length genomes, where every candidate has the
same loci. That is the case every measured row in the research record was run
against.
"""

from __future__ import annotations

import random
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_evolve import ObjectiveSpec
from agent_evolve.api import optimize
from agent_evolve.policies.genetic import Locus, crossover, loci_of


def test_loci_count_is_per_candidate_not_per_problem():
    short = {"selection": [5]}
    long = {"selection": [0, 5, 9]}
    assert len(loci_of(short)) == 1
    assert len(loci_of(long)) == 3


def test_crossover_keeps_parent_a_where_the_donor_has_no_such_locus():
    a = {"selection": [1, 2, 3]}
    b = {"selection": [9]}
    # Ask for every locus from the donor. Only locus 0 exists there.
    child = crossover(a, b, mask=[True, True, True])
    assert child == {"selection": [9, 2, 3]}
    # Neither parent is mutated.
    assert a == {"selection": [1, 2, 3]}
    assert b == {"selection": [9]}


def test_crossover_still_refuses_a_mask_of_the_wrong_length():
    """The strict check is a caller contract and must survive the fix."""
    with pytest.raises(ValueError, match="mask has 1 bits but the candidate has 3 loci"):
        crossover({"selection": [1, 2, 3]}, {"selection": [9]}, mask=[True])


def test_crossover_on_equal_shapes_is_unchanged():
    a = {"x": 1, "y": 2, "z": 3}
    b = {"x": 9, "y": 8, "z": 7}
    assert crossover(a, b, mask=[False, True, False]) == {"x": 1, "y": 8, "z": 3}
    # And the maskless path still consumes exactly one bit per locus, which is
    # what keeps the credential-free default stream byte-identical.
    left = crossover(a, b, rng=random.Random(4))
    right = crossover(a, b, rng=random.Random(4))
    assert left == right


def test_crossover_donor_membership_is_by_locus_not_by_position():
    """A field the donor lacks entirely is also simply not inheritable."""
    a = {"p": 1, "q": [7, 8]}
    b = {"q": [5, 6]}
    assert Locus("p") in loci_of(a)
    assert Locus("p") not in loci_of(b)
    assert crossover(a, b, mask=[True, True, True]) == {"p": 1, "q": [5, 6]}


class _RaggedConfig(BaseModel):
    picks: list[Literal[1, 2, 3, 4]]


class _RaggedSeedProblem:
    """The knapsack example's shape, reduced to the part that crashed.

    Seeds of different genome length, and a field the schema declares as an
    enum so the loop has something it can legally resample.
    """

    candidate_model = _RaggedConfig
    objectives = [ObjectiveSpec("total", "max"), ObjectiveSpec("count", "min")]

    def seeds(self):
        return [{"picks": [1]}, {"picks": [1, 2, 3]}]

    def evaluate(self, artifact):
        picks = list(artifact["picks"])
        return {"total": float(sum(picks)), "count": float(len(picks))}


def test_the_loop_runs_when_the_seeds_have_different_genome_lengths():
    result = optimize(_RaggedSeedProblem(), budget=16, proposer="random", seed=3)
    assert result.evaluations >= 2
    assert result.pareto_front
    # Every configuration the loop produced is a legal instance of the schema:
    # ragged recombination must not invent a value or a length the model refuses.
    for candidate in result.pareto_front:
        _RaggedConfig.model_validate(candidate.configuration)


def test_the_shipped_knapsack_example_runs_credential_free():
    """The README's reference integration, exercised rather than described."""
    problem_def = pytest.importorskip(
        "examples.knapsack.problem_def",
        reason="run from the repository root to exercise the shipped examples",
    )
    result = optimize(problem_def.problem, budget=24, proposer="random", seed=7)
    assert result.pareto_front
    assert result.best is not None
    for candidate in result.pareto_front:
        problem_def.CandidateConfig.model_validate(candidate.configuration)

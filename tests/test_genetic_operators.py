"""Genetic operators must be workload-agnostic and must not invent domains."""

from __future__ import annotations

import random
from typing import Literal

import pytest
from pydantic import BaseModel

from agent_evolve.policies.genetic import (
    Locus,
    crossover,
    loci_of,
    locus_domain,
    mutate,
    read_locus,
    tournament,
    truncation_survival,
    write_locus,
)


class _Candidate(BaseModel):
    """A deliberately unremarkable model: one sequence, one scalar, one free field."""

    sequence: list[Literal["a", "b", "c"]]
    width: Literal[2, 4, 8]
    label: str


def test_sequence_fields_expand_element_wise() -> None:
    # A genome whose only locus is "the whole list" cannot be recombined, which
    # is the degenerate case that leaves whole-artifact rewrite as the only move.
    config = {"sequence": ["a", "b", "c"], "width": 4, "label": "x"}
    assert loci_of(config) == (
        Locus("sequence", 0),
        Locus("sequence", 1),
        Locus("sequence", 2),
        Locus("width"),
        Locus("label"),
    )


def test_write_locus_does_not_mutate_its_input() -> None:
    config = {"sequence": ["a", "b"], "width": 2, "label": "x"}
    out = write_locus(config, Locus("sequence", 0), "c")
    assert out["sequence"] == ["c", "b"]
    assert config["sequence"] == ["a", "b"], "input was mutated in place"


def test_crossover_takes_each_locus_from_the_masked_parent() -> None:
    a = {"sequence": ["a", "a", "a"], "width": 2, "label": "A"}
    b = {"sequence": ["b", "b", "b"], "width": 8, "label": "B"}
    child = crossover(a, b, mask=[False, True, False, True, False])
    assert child["sequence"] == ["a", "b", "a"]
    assert child["width"] == 8
    assert child["label"] == "A"


def test_crossover_rejects_a_mask_of_the_wrong_length() -> None:
    a = {"sequence": ["a", "a"], "width": 2, "label": "A"}
    with pytest.raises(ValueError, match="mask has 2 bits"):
        crossover(a, a, mask=[True, False])


def test_crossover_of_identical_parents_is_that_parent() -> None:
    a = {"sequence": ["a", "b"], "width": 4, "label": "x"}
    assert crossover(a, a, rng=random.Random(0)) == a


def test_locus_domain_reads_the_problems_own_schema() -> None:
    assert set(locus_domain(_Candidate, Locus("sequence", 0))) == {"a", "b", "c"}
    assert set(locus_domain(_Candidate, Locus("width"))) == {2, 4, 8}


def test_locus_domain_is_empty_when_the_schema_declares_no_finite_set() -> None:
    # `label` is an unconstrained string. Returning () means mutate() leaves it
    # alone rather than inventing values the problem never declared.
    assert locus_domain(_Candidate, Locus("label")) == ()


def test_mutate_leaves_loci_whose_domain_is_undeclared() -> None:
    config = {"sequence": ["a", "a"], "width": 2, "label": "keep me"}
    out = mutate(config, _Candidate, loci=[Locus("label")], rng=random.Random(0))
    assert out["label"] == "keep me"


def test_mutate_changes_exactly_the_named_loci_and_actually_changes_them() -> None:
    config = {"sequence": ["a", "a", "a"], "width": 2, "label": "x"}
    out = mutate(config, _Candidate, loci=[Locus("sequence", 1)],
                 rng=random.Random(1))
    assert out["sequence"][0] == "a" and out["sequence"][2] == "a"
    assert out["sequence"][1] in {"b", "c"}, "a mutation that changes nothing is not one"


def test_mutate_without_a_model_is_a_no_op_rather_than_a_crash() -> None:
    config = {"sequence": ["a"], "width": 2, "label": "x"}
    assert mutate(config, None, rng=random.Random(0)) == config


def test_tournament_returns_the_better_of_the_contenders() -> None:
    population = [({"i": 0}, 5.0), ({"i": 1}, 1.0), ({"i": 2}, 9.0)]
    picked = tournament(population, size=3, rng=random.Random(0))
    assert picked == {"i": 1}


def test_tournament_on_an_empty_population_says_so() -> None:
    with pytest.raises(ValueError, match="empty population"):
        tournament([], rng=random.Random(0))


def test_survival_deduplicates_before_truncating() -> None:
    # Without dedup a population collapses onto copies of one genome and
    # recombination stops producing anything new.
    population = [
        ({"g": "same"}, 1.0),
        ({"g": "same"}, 2.0),
        ({"g": "other"}, 3.0),
        ({"g": "third"}, 4.0),
    ]
    kept = truncation_survival(population, keep=2, key_of=lambda c: c["g"])
    assert [c for c, _ in kept] == [{"g": "same"}, {"g": "other"}]
    assert kept[0][1] == 1.0, "dedup kept the worse copy of a duplicate genome"


def test_operators_never_reference_a_workload() -> None:
    # The whole point: this module is generic. If a workload noun appears here,
    # the abstraction has leaked and the next domain will need a special case.
    import pathlib

    import agent_evolve.policies.genetic as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8").lower()
    for noun in ("abc", "boils", "timeloop", "lut", "circuit", "spice", "kernel"):
        assert f" {noun} " not in source and f"_{noun}" not in source, (
            f"workload vocabulary {noun!r} leaked into a generic operator module"
        )

"""Bounded numeric loci are searchable; ``const`` is a one-value domain.

Before 2026-08-19 ``_declared_domain`` returned ``()`` for anything that was not
an enum or a bool, so every bounded ``int``/``float`` field was frozen at the
template's value: 12 of 13 axes on a hyperparameter benchmark were identical
across 64 draws, and the shipped knapsack example searched nothing. A range is
a declaration; reading it as one is not inventing a domain, it is refusing to
throw one away.

The projection is the whole claim, so it is tested as one: what the schema says,
what comes back, and -- t11 to t13 -- that the operators, the prior narrowing
and the report downstream all move on it.
"""

from __future__ import annotations

import random
from typing import Annotated, Literal, Optional

import pytest
from pydantic import BaseModel, Field

from agent_evolve.policies.genetic import (
    DomainRestriction,
    Locus,
    locus_domain,
    locus_is_projected,
    mutate,
    uniform_candidate,
)


class _Numeric(BaseModel):
    """One field per reading the projection has to make."""

    small_int: int = Field(1, ge=1, le=8)                       # t1
    wide_int: int = Field(0, ge=0, le=1000)                     # t2
    open_int: int = Field(1, gt=0, lt=10)                       # t3
    stepped_int: int = Field(0, ge=0, le=100, multiple_of=25)   # t4
    unit_float: float = Field(0.5, ge=0.0, le=1.0)              # t5
    open_float: float = Field(0.5, gt=0.0, lt=1.0)              # t6
    one_literal: Literal["x"] = "x"                             # t7
    pinned: int = Field(3, json_schema_extra={"const": 3})      # t7
    maybe_int: Optional[int] = Field(None, ge=1, le=4)          # t8
    coarse: float = Field(                                      # t9
        0.5, ge=0.0, le=1.0,
        json_schema_extra={"agent_evolve": {"grid": 5}},
    )
    free_text: str = "z"                                        # t10
    free_int: int = 0                                           # t10
    indices: list[Annotated[int, Field(ge=0, le=3)]] = Field(default_factory=list)


def _domain(name: str, index: int | None = None) -> tuple:
    return locus_domain(_Numeric, Locus(name, index))


# -- t1..t4: integers -------------------------------------------------------


def test_a_short_integer_range_is_enumerated_exactly() -> None:
    assert _domain("small_int") == tuple(range(1, 9))
    assert locus_is_projected(_Numeric, Locus("small_int"))


def test_a_wide_integer_range_is_projected_onto_a_bounded_grid() -> None:
    values = _domain("wide_int")
    assert len(values) == 64
    assert values[0] == 0 and values[-1] == 1000, "the endpoints are legal values"
    assert all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    assert list(values) == sorted(set(values)), "strictly increasing, deduplicated"


def test_exclusive_integer_bounds_tighten_by_one() -> None:
    assert _domain("open_int") == tuple(range(1, 10))


def test_multiple_of_enumerates_multiples_the_schema_would_accept() -> None:
    assert _domain("stepped_int") == (0, 25, 50, 75, 100)


# -- t5..t6: numbers --------------------------------------------------------


def test_a_bounded_float_is_a_sixteen_point_inclusive_grid() -> None:
    values = _domain("unit_float")
    assert len(values) == 16
    assert values[0] == 0.0 and values[-1] == 1.0
    assert list(values) == sorted(set(values))
    assert all(repr(v) == repr(float(f"{v:.10g}")) for v in values), (
        "grid floats must render the same everywhere they are shown")


def test_excluded_float_endpoints_are_not_emitted() -> None:
    values = _domain("open_float")
    assert len(values) == 16
    assert 0.0 not in values and 1.0 not in values
    assert values[0] > 0.0 and values[-1] < 1.0
    assert list(values) == sorted(set(values))


# -- t7..t10: const, optional, override, and the honest empty ---------------


def test_a_const_node_is_a_one_value_domain() -> None:
    # A Literal of length one serializes to {"const": ...}, not an enum. Read as
    # nothing, it froze its locus and every report called it undeclared.
    assert _domain("one_literal") == ("x",)
    assert _domain("pinned") == (3,)
    assert not locus_is_projected(_Numeric, Locus("one_literal")), (
        "a constant is declared outright, not projected off a range")


def test_an_optional_bounded_integer_projects_without_none() -> None:
    values = _domain("maybe_int")
    assert values == (1, 2, 3, 4)
    assert None not in values


def test_a_field_can_declare_its_own_grid() -> None:
    assert _domain("coarse") == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_unbounded_fields_still_declare_nothing() -> None:
    assert _domain("free_text") == ()
    assert _domain("free_int") == ()
    assert not locus_is_projected(_Numeric, Locus("free_text"))


def test_sequence_elements_project_like_scalars() -> None:
    assert _domain("indices", 0) == (0, 1, 2, 3)
    assert _domain("indices") == (0, 1, 2, 3), "the shared per-element vocabulary"


# -- t11..t13: the operators and the priors actually move on it -------------


def test_uniform_candidate_varies_a_bounded_float_locus() -> None:
    class _OneFloat(BaseModel):
        rate: float = Field(0.5, ge=0.0, le=1.0)

    rng = random.Random(0)
    template = {"rate": 0.5}
    drawn = {uniform_candidate(template, _OneFloat, rng=rng)["rate"] for _ in range(32)}
    assert len(drawn) >= 8, (
        "a bounded float locus that never moves is the frozen-axis defect")
    assert drawn <= set(locus_domain(_OneFloat, Locus("rate")))


def test_mutate_resamples_a_bounded_integer_locus() -> None:
    class _OneInt(BaseModel):
        depth: int = Field(4, ge=1, le=16)

    config = {"depth": 4}
    changed = [
        mutate(config, _OneInt, loci=[Locus("depth")], rng=random.Random(s))["depth"]
        for s in range(8)
    ]
    assert all(v != 4 for v in changed), "mutate must not return the current value"
    assert all(1 <= v <= 16 for v in changed)


def test_a_restriction_narrows_a_projected_domain_and_counts_a_miss() -> None:
    class _Restricted(BaseModel):
        depth: int = Field(4, ge=1, le=8)

    keep = DomainRestriction({"depth": [2, 4, 99]})
    assert locus_domain(_Restricted, Locus("depth"), restriction=keep) == (2, 4)
    assert keep.misses == [], "a partial overlap is a narrowing, not a miss"

    disjoint = DomainRestriction({"depth": [100, 200]})
    assert locus_domain(_Restricted, Locus("depth"), restriction=disjoint) == \
        tuple(range(1, 9)), "an empty intersection returns the declared domain"
    assert disjoint.misses == ["depth"]


# -- the diagnostic says which domains are projected ------------------------


def test_the_check_report_marks_projected_axes_with_their_point_count() -> None:
    from agent_evolve.policies.check import check
    from test_public_contract import Complete

    report = check(Complete(), 8, probe=8, seed=0)
    assert report.undeclared_loci == ()
    assert [(d.locus, d.domain_size, d.projected) for d in report.loci] == [
        ("x", 10, True), ("y", 10, True)]
    assert "x:10(projected)" in report.render()


@pytest.mark.parametrize(
    "node",
    [
        {"type": "integer", "minimum": 5, "maximum": 1},   # empty interval
        {"type": "integer", "minimum": 0},                 # one-sided
        {"type": "number", "exclusiveMinimum": 1.0, "exclusiveMaximum": 1.0},
        {"type": "integer", "minimum": 0, "maximum": 9, "multipleOf": 2.5},
    ],
)
def test_a_range_with_no_finite_reading_declares_nothing(node: dict) -> None:
    from agent_evolve.policies.genetic import _node_domain

    assert _node_domain(node, None)[0] == ()

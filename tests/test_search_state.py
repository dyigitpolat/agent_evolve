"""The chooser's context must be derived, separable, and free of invented weights."""

from __future__ import annotations

from agent_evolve.policies.genetic import Locus
from agent_evolve.policies.search_state import (
    SearchState,
    StateChannels,
    locus_table,
    render_history,
    render_locus_table,
)

_OBJ = ("cost", "depth")


def _evaluated():
    return [
        ({"g": ["a", "x"]}, {"cost": 10.0, "depth": 3.0}),
        ({"g": ["a", "y"]}, {"cost": 20.0, "depth": 5.0}),
        ({"g": ["b", "x"]}, {"cost": 6.0, "depth": 4.0}),
    ]


def test_table_counts_each_value_at_each_locus() -> None:
    table = locus_table(_evaluated(), _OBJ)
    first = {s.value: s.count for s in table[Locus("g", 0)]}
    assert first == {"a": 2, "b": 1}
    second = {s.value: s.count for s in table[Locus("g", 1)]}
    assert second == {"x": 2, "y": 1}


def test_objectives_are_kept_separate_never_scalarised() -> None:
    # A scalarisation needs weights nobody declared, and an undeclared weight is
    # the kind of hidden choice that has silently decided results here before.
    stat = next(s for s in locus_table(_evaluated(), _OBJ)[Locus("g", 0)]
                if s.value == "a")
    assert stat.mean == {"cost": 15.0, "depth": 4.0}
    assert stat.best == {"cost": 10.0, "depth": 3.0}
    assert not hasattr(stat, "score")


def test_saturation_is_computed_and_stated_not_left_to_the_reader() -> None:
    evaluated = [({"g": ["a"]}, {"cost": 1.0}), ({"g": ["a"]}, {"cost": 2.0})]
    text = render_locus_table(locus_table(evaluated, ("cost",)), ("cost",))
    assert "SATURATED" in text, "a locus with one value across every observation"


def test_a_single_observation_is_marked_barely_explored() -> None:
    evaluated = [({"g": ["a"]}, {"cost": 1.0})]
    text = render_locus_table(locus_table(evaluated, ("cost",)), ("cost",))
    assert "barely explored" in text


def test_empty_history_and_table_do_not_crash() -> None:
    assert "nothing evaluated" in render_locus_table({}, _OBJ)
    assert "no generations" in render_history([])


def test_history_surfaces_choices_that_were_filled_at_random() -> None:
    # A guided arm whose choices were silently topped up is a run whose result
    # cannot be attributed; the chooser should see that it happened.
    text = render_history([{"gen": 3, "valid_count": 6,
                            "choices_filled_at_random": 2}])
    assert "filled at random" in text


def test_channels_are_independently_switchable() -> None:
    state = SearchState(
        channels=StateChannels.only("locus_measurements"),
        evaluated=_evaluated(),
        history=[{"gen": 1, "valid_count": 3}],
        side_information=["compiler said no"],
        rules=["prefer short recipes"],
    )
    text = state.render(_OBJ)
    assert "PER-LOCUS" in text
    assert "RECENT GENERATIONS" not in text
    assert "compiler said no" not in text
    assert "prefer short recipes" not in text


def test_a_disabled_channel_is_absent_not_merely_empty() -> None:
    # An empty section still tells the reader the channel exists, which is a
    # difference an ablation must not carry.
    off = SearchState(channels=StateChannels.none(), evaluated=_evaluated())
    assert off.render(_OBJ) == ""


def test_all_channels_on_renders_every_section() -> None:
    state = SearchState(
        evaluated=_evaluated(),
        history=[{"gen": 1, "valid_count": 3}],
        side_information=["timeout on candidate 4"],
        rules=["depth improves when position 0 is b"],
    )
    text = state.render(_OBJ)
    for marker in ("PER-LOCUS", "EVALUATOR DIAGNOSTICS", "RECENT GENERATIONS",
                   "WHAT HAS HELD SO FAR"):
        assert marker in text


def test_only_rejects_an_unknown_channel_by_name() -> None:
    import pytest

    with pytest.raises(ValueError, match="unknown channel"):
        StateChannels.only("vibes")


def test_state_module_names_no_workload() -> None:
    import pathlib

    import agent_evolve.policies.search_state as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8").lower()
    for noun in ("abc", "boils", "timeloop", "lut", "circuit", "spice"):
        assert f" {noun} " not in source and f"_{noun}" not in source

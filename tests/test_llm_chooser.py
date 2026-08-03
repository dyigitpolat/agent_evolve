"""The model chooses operators. It must not be able to author a candidate."""

from __future__ import annotations

import json

from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.policies.llm_chooser import (
    ChooserTelemetry,
    llm_chooser,
    parse_choices,
    render_population,
)

_SPECS = (ObjectiveSpec(name="ones", goal="max"),)
_POP = [({"genome": [0, 0, 0, 0]}, 1.0), ({"genome": [1, 1, 1, 1]}, 0.0)]


def _fresh() -> ChooserTelemetry:
    return ChooserTelemetry()


def test_a_well_formed_reply_is_accepted() -> None:
    text = '[{"parent_a": 0, "parent_b": 1, "mask": [0,1,0,1]}]'
    t = _fresh()
    out = parse_choices(text, n_loci=4, population_size=2, telemetry=t)
    assert len(out) == 1
    assert out[0].parent_a == 0 and out[0].parent_b == 1
    assert out[0].mask == (False, True, False, True)
    assert t.accepted == 1 and t.wrote_candidate == 0


def test_prose_around_the_json_is_tolerated() -> None:
    text = 'Sure! Here you go:\n[{"parent_a": 1, "parent_b": 0, "mask": [1,1,0,0]}]\nHope that helps.'
    out = parse_choices(text, n_loci=4, population_size=2, telemetry=_fresh())
    assert len(out) == 1


def test_a_reply_that_writes_a_candidate_is_rejected_and_counted() -> None:
    # This is the failure that would collapse operator guidance into
    # whole-artifact rewrite -- the paradigm the claim is defined against.
    text = '[{"genome": [1,0,1,0], "parent_a": 0, "parent_b": 1}]'
    t = _fresh()
    out = parse_choices(text, n_loci=4, population_size=2, telemetry=t)
    assert out == []
    assert t.wrote_candidate == 1, "a candidate-shaped reply was not counted"


def test_a_nested_structure_counts_as_a_candidate() -> None:
    text = '[{"parent_a": 0, "parent_b": 1, "mask": [0,0,0,0], "child": {"genome": [1,1,1,1]}}]'
    t = _fresh()
    assert parse_choices(text, n_loci=4, population_size=2, telemetry=t) == []
    assert t.wrote_candidate == 1


def test_out_of_range_parents_are_rejected_not_clamped() -> None:
    # Clamping would quietly turn an invalid choice into a valid-looking one and
    # credit the model for a decision it did not make.
    text = '[{"parent_a": 0, "parent_b": 99, "mask": [0,0,0,0]}]'
    t = _fresh()
    assert parse_choices(text, n_loci=4, population_size=2, telemetry=t) == []
    assert t.out_of_range == 1


def test_a_wrong_length_mask_is_rejected_and_counted() -> None:
    text = '[{"parent_a": 0, "parent_b": 1, "mask": [0,1]}]'
    t = _fresh()
    assert parse_choices(text, n_loci=4, population_size=2, telemetry=t) == []
    assert t.out_of_range == 1


def test_unparseable_output_is_counted_rather_than_raised() -> None:
    t = _fresh()
    assert parse_choices("I cannot help with that.", n_loci=4,
                         population_size=2, telemetry=t) == []
    assert t.unparseable == 1


def test_chooser_reports_a_shortfall_instead_of_hiding_it() -> None:
    # A run that quietly substitutes random choices for model choices is a run
    # whose result cannot be attributed.
    seen: list[tuple[int, int]] = []
    chooser = llm_chooser(lambda _p: "no json here", objectives=_SPECS,
                          budget=10, on_shortfall=lambda got, want: seen.append((got, want)))
    out = chooser(_POP, 3)
    assert out == []
    assert seen == [(0, 3)]


def test_provider_errors_are_counted_not_propagated() -> None:
    def boom(_prompt: str) -> str:
        raise RuntimeError("provider down")

    t = ChooserTelemetry()
    chooser = llm_chooser(boom, objectives=_SPECS, budget=10, telemetry=t)
    assert chooser(_POP, 2) == []
    assert t.errors == 1


def test_the_prompt_carries_the_population_and_forbids_authoring() -> None:
    captured: list[str] = []

    def capture(prompt: str) -> str:
        captured.append(prompt)
        return '[{"parent_a": 0, "parent_b": 1, "mask": [0,0,0,0]}]'

    llm_chooser(capture, objectives=_SPECS, budget=10)(_POP, 1)
    prompt = captured[0]
    assert "genome" in prompt, "the population was not shown"
    assert "Do NOT write candidates" in prompt
    assert "mask" in prompt.lower()


def test_rendered_population_is_lossless_json() -> None:
    text = render_population(_POP, _SPECS)
    assert json.dumps({"genome": [0, 0, 0, 0]}, sort_keys=True) in text
    assert "rank 1" in text


def test_chooser_module_names_no_workload() -> None:
    import pathlib

    import agent_evolve.policies.llm_chooser as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8").lower()
    for noun in ("abc", "boils", "timeloop", "lut", "circuit", "spice"):
        assert f" {noun} " not in source and f"_{noun}" not in source

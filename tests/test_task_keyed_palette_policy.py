"""Focused contracts for task-keyed path and atomic-palette selection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.domain.variation_space import AtomicEditOption
from agent_evolve.policies.selection.task_keyed_palette import (
    PathFamilyExposure,
    PathSelectionMode,
    TaskKeyedPalettePolicy,
)


def _path(index: int) -> JsonPath:
    return JsonPath((ObjectKey("sequence"), ArrayIndex(index)))


def _option(index: int, suffix: str, family: str) -> AtomicEditOption:
    return AtomicEditOption(
        option_id=f"fixture.p{index}.{suffix}",
        path=_path(index),
        replacement=f"replacement_{suffix}",
        family=family,
        metadata=(("source", f"source-{index}-{suffix}"),),
    )


def _catalog() -> tuple[AtomicEditOption, ...]:
    return tuple(
        _option(index, suffix, family)
        for index in range(3)
        for suffix, family in (
            ("a", "family_a"),
            ("b", "family_b"),
            ("c", "family_c"),
            ("d", "family_d"),
        )
    )


def test_unforced_path_uses_minimum_exposure_then_task_hash_not_input_order() -> None:
    catalog = _catalog()
    exposures = (
        PathFamilyExposure(_path(0), "family_a", 5),
        PathFamilyExposure(_path(1), "family_a", 1),
        PathFamilyExposure(_path(2), "family_a", 1),
    )
    policy = TaskKeyedPalettePolicy(seed=781)

    forward = policy.select(
        task_key="g1-uncertainty",
        options=catalog,
        palette_size=3,
        exposures=exposures,
    )
    reverse = policy.select(
        task_key="g1-uncertainty",
        options=tuple(reversed(catalog)),
        palette_size=3,
        exposures=tuple(reversed(exposures)),
    )

    assert forward == reverse
    assert forward.decision_sha256 == reverse.decision_sha256
    assert forward.path_selection_mode is (
        PathSelectionMode.MINIMUM_EXPOSURE_TASK_KEYED
    )
    eligible_minimum = min(
        (row for row in forward.path_rows if row.eligible_for_choice),
        key=lambda row: (row.path_exposure, row.task_order_sha256),
    )
    assert forward.chosen_path == eligible_minimum.path
    assert forward.chosen_path in {_path(1), _path(2)}
    assert {option.path for option in forward.palette} == {forward.chosen_path}
    assert len(forward.palette) == 3


def test_requirements_fix_one_path_but_never_control_provider_order() -> None:
    catalog = _catalog()
    required = ("fixture.p2.d", "fixture.p2.a")
    decision = TaskKeyedPalettePolicy(seed=13).select(
        task_key="g1-area-card-pair",
        options=catalog,
        palette_size=3,
        exposures=(PathFamilyExposure(_path(2), "family_d", 99),),
        required_option_ids=required,
    )

    assert decision.path_selection_mode is PathSelectionMode.REQUIRED_OPTIONS
    assert decision.chosen_path == _path(2)
    assert decision.required_option_ids == tuple(sorted(required))
    assert set(required).issubset({option.option_id for option in decision.palette})
    assert tuple(option.option_id for option in decision.palette) == tuple(
        row.option.option_id
        for row in sorted(
            (
                row
                for row in decision.option_rows
                if row.palette_position is not None
            ),
            key=lambda row: row.palette_position,
        )
    )
    assert tuple(option.option_id for option in decision.palette) == tuple(
        row.option.option_id
        for row in sorted(
            (
                row
                for row in decision.option_rows
                if row.palette_position is not None
            ),
            key=lambda row: (row.presentation_order_sha256, row.option.option_id),
        )
    )

    trace = decision.to_trace_record()
    assert trace["required_option_ids"] == list(sorted(required))
    assert trace["palette_option_ids"] == [
        option.option_id for option in decision.palette
    ]
    assert len(trace["path_rows"]) == 3
    assert len(trace["option_rows"]) == len(catalog)
    assert trace["decision_sha256"] == decision.decision_sha256


def test_forced_path_fill_uses_cell_then_family_exposure_and_task_tie_break() -> None:
    catalog = _catalog()
    exposures = (
        # Equal zero cell exposure at p0 for A/B; the history on another path
        # makes B the lower-exposure family and therefore the first fill row.
        PathFamilyExposure(_path(1), "family_a", 7),
        PathFamilyExposure(_path(1), "family_b", 2),
        PathFamilyExposure(_path(0), "family_c", 1),
        PathFamilyExposure(_path(0), "family_d", 4),
    )
    decision = TaskKeyedPalettePolicy(seed=9).select(
        task_key="g1-depth-controlled",
        options=tuple(reversed(catalog)),
        palette_size=2,
        exposures=exposures,
        path=_path(0),
        required_option_ids=("fixture.p0.d",),
    )

    assert decision.path_selection_mode is PathSelectionMode.FORCED
    assert decision.chosen_path == _path(0)
    selected = {option.option_id for option in decision.palette}
    assert selected == {"fixture.p0.b", "fixture.p0.d"}
    rows = {row.option.option_id: row for row in decision.option_rows}
    assert rows["fixture.p0.a"].cell_exposure == 0
    assert rows["fixture.p0.a"].family_exposure == 7
    assert rows["fixture.p0.b"].cell_exposure == 0
    assert rows["fixture.p0.b"].family_exposure == 2
    assert rows["fixture.p0.d"].required is True


def test_family_cap_is_an_engine_constraint_for_requirements_fill_and_path_feasibility() -> None:
    crowded = (
        _option(0, "a", "family_a"),
        _option(0, "b", "family_a"),
        _option(0, "c", "family_b"),
        _option(0, "d", "family_b"),
    )
    diverse = tuple(
        _option(1, suffix, family)
        for suffix, family in (
            ("a", "family_a"),
            ("b", "family_b"),
            ("c", "family_c"),
        )
    )
    policy = TaskKeyedPalettePolicy(seed=91)
    decision = policy.select(
        task_key="distinct-families",
        options=(*crowded, *diverse),
        palette_size=3,
        max_options_per_family=1,
    )

    assert decision.chosen_path == _path(1)
    assert decision.max_options_per_family == 1
    assert len({option.family for option in decision.palette}) == 3
    rows = {row.path: row for row in decision.path_rows}
    assert rows[_path(0)].option_count == 4
    assert rows[_path(0)].family_capacity == 2
    assert rows[_path(0)].feasible_for_size is False
    assert rows[_path(1)].family_capacity == 3
    assert decision.to_trace_record()["max_options_per_family"] == 1

    with pytest.raises(ValueError, match="max_options_per_family"):
        policy.select(
            task_key="invalid-required-family",
            options=(*crowded, *diverse),
            palette_size=3,
            max_options_per_family=1,
            path=_path(0),
            required_option_ids=("fixture.p0.a", "fixture.p0.b"),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"required_option_ids": ("fixture.p0.a",) * 2}, "duplicates"),
        ({"required_option_ids": ("fixture.p0.a", "fixture.p1.b")}, "share"),
        ({"required_option_ids": ("fixture.missing",)}, "absent"),
        (
            {
                "path": _path(0),
                "required_option_ids": ("fixture.p1.a",),
            },
            "forced path",
        ),
        (
            {
                "palette_size": 1,
                "required_option_ids": ("fixture.p0.a", "fixture.p0.b"),
            },
            "exceed",
        ),
    ],
)
def test_required_option_constraints_fail_closed(kwargs: dict[str, object], match: str) -> None:
    arguments: dict[str, object] = {
        "task_key": "fail-closed",
        "options": _catalog(),
        "palette_size": 3,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=match):
        TaskKeyedPalettePolicy(seed=1).select(**arguments)  # type: ignore[arg-type]


def test_catalog_and_exposure_ambiguity_fail_closed() -> None:
    catalog = _catalog()
    duplicate_id = AtomicEditOption(
        option_id=catalog[0].option_id,
        path=catalog[1].path,
        replacement="different",
        family="family_z",
    )
    duplicate_replacement = AtomicEditOption(
        option_id="fixture.p0.duplicate",
        path=catalog[0].path,
        replacement=catalog[0].replacement,
        family="family_z",
    )
    policy = TaskKeyedPalettePolicy(seed=1)
    with pytest.raises(ValueError, match="option_id"):
        policy.select(
            task_key="duplicate-id",
            options=(*catalog, duplicate_id),
            palette_size=2,
        )
    with pytest.raises(ValueError, match="duplicate replacements"):
        policy.select(
            task_key="duplicate-value",
            options=(*catalog, duplicate_replacement),
            palette_size=2,
        )
    repeated_exposure = PathFamilyExposure(_path(0), "family_a", 0)
    with pytest.raises(ValueError, match="exposure cells"):
        policy.select(
            task_key="duplicate-exposure",
            options=catalog,
            palette_size=2,
            exposures=(repeated_exposure, repeated_exposure),
        )


def test_decision_is_frozen_and_rejects_forged_palette_or_rows() -> None:
    decision = TaskKeyedPalettePolicy(seed=4).select(
        task_key="immutable",
        options=_catalog(),
        palette_size=3,
        path=_path(0),
    )
    with pytest.raises(FrozenInstanceError):
        decision.palette_size = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="palette"):
        replace(decision, palette=tuple(reversed(decision.palette)))
    with pytest.raises(ValueError, match="selection facts"):
        replace(decision, option_rows=decision.option_rows[:-1])

from __future__ import annotations

import copy

import pytest

from agent_evolve.policies.variation.crossover_inheritance import (
    CrossoverInheritanceClaim,
    CrossoverInheritanceSource,
    materialize_crossover_inheritance,
)


_LEFT = {
    "left_branch": {"gene": 1},
    "right_branch": {"gene": 0},
    "shared": {
        "map": {"keep": 1, "optional": None},
        "list": [1, 2, 3],
    },
}
_RIGHT = {
    "left_branch": {"gene": 0},
    "right_branch": {"gene": 2},
    "shared": copy.deepcopy(_LEFT["shared"]),
}
_INHERITANCE_CLAIMS = (
    CrossoverInheritanceClaim(
        "$.left_branch",
        CrossoverInheritanceSource.LEFT,
    ),
    CrossoverInheritanceClaim(
        "$.right_branch",
        CrossoverInheritanceSource.RIGHT,
    ),
)


def _base_child() -> dict[str, object]:
    return {
        "left_branch": copy.deepcopy(_LEFT["left_branch"]),
        "right_branch": copy.deepcopy(_RIGHT["right_branch"]),
        "shared": copy.deepcopy(_LEFT["shared"]),
    }


def test_unclaimed_identically_shared_top_level_deletion_fails_closed() -> None:
    child = _base_child()
    del child["shared"]

    with pytest.raises(
        ValueError,
        match="omits a shared component without synthesized container attribution",
    ):
        materialize_crossover_inheritance(
            left=_LEFT,
            right=_RIGHT,
            draft=child,
            claims=_INHERITANCE_CLAIMS,
        )


@pytest.mark.parametrize(
    ("edit", "synthesized_path"),
    [
        ("delete_optional_map_member", "$.shared.map"),
        ("add_map_member", "$.shared.map.added"),
        ("change_shared_scalar", "$.shared.map.keep"),
        ("delete_list_member", "$.shared.list"),
    ],
)
def test_shared_structural_edits_require_explicit_synthesis(
    edit: str,
    synthesized_path: str,
) -> None:
    child = _base_child()
    shared = child["shared"]
    assert type(shared) is dict
    shared_map = shared["map"]
    assert type(shared_map) is dict
    shared_list = shared["list"]
    assert type(shared_list) is list
    if edit == "delete_optional_map_member":
        del shared_map["optional"]
    elif edit == "add_map_member":
        shared_map["added"] = 7
    elif edit == "change_shared_scalar":
        shared_map["keep"] = 9
    elif edit == "delete_list_member":
        shared_list.pop()
    else:  # pragma: no cover - closed parameter table.
        raise AssertionError("unknown structural edit fixture")

    with pytest.raises(ValueError):
        materialize_crossover_inheritance(
            left=_LEFT,
            right=_RIGHT,
            draft=child,
            claims=_INHERITANCE_CLAIMS,
        )

    materialized = materialize_crossover_inheritance(
        left=_LEFT,
        right=_RIGHT,
        draft=child,
        claims=_INHERITANCE_CLAIMS
        + (
            CrossoverInheritanceClaim(
                synthesized_path,
                CrossoverInheritanceSource.SYNTHESIZED,
            ),
        ),
    )

    assert [item.path for item in materialized.synthesized_paths] == [
        synthesized_path
    ]
    assert len(materialized.receipt_sha256) == 64


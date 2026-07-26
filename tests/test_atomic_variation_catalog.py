"""Focused offline contracts for generic and BOiLS atomic option catalogs."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.patch import ArrayIndex, JsonPath, ObjectKey
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.domain.variation_space import AtomicEditOption
from agent_evolve.ports.variation_catalog import AtomicVariationCatalog
from examples.benchmarks.boils_abc.actions import (
    ACTION_COMMANDS,
    ACTION_IDS,
    DEFAULT_ACTION_SEQUENCE,
    SEQUENCE_LENGTH,
)
from examples.benchmarks.boils_abc.variation_catalog import (
    ACTION_DEFINITION_SHA256,
    ACTION_FAMILIES,
    CATALOG_SOURCE_SHA256,
    BoilsAtomicVariationCatalog,
)


def _candidate(
    sequence: tuple[str, ...] = tuple(DEFAULT_ACTION_SEQUENCE),
    *,
    suffix: str = "parent",
) -> EvolutionCandidate:
    configuration = freeze_json({"sequence": list(sequence)})
    digest = typed_json_sha256(configuration)
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(f"candidate_{suffix}"),
            configuration_hash=digest,
            configuration_artifact_hash=digest,
            proposal_sequence=0,
        ),
        configuration=configuration,
        objectives=(("total_lut_count", 1.0), ("total_levels", 1.0)),
        valid=True,
        generation=0,
        label=suffix,
    )


def test_atomic_edit_option_is_frozen_typed_and_fully_identity_bound() -> None:
    path = JsonPath((ObjectKey("sequence"), ArrayIndex(3)))
    option = AtomicEditOption(
        option_id="fixture.sequence_03.rewrite",
        path=path,
        replacement="rewrite",
        family="aig_rewrite",
        metadata=(("command", "rewrite"),),
    )
    same = AtomicEditOption(
        option_id="fixture.sequence_03.rewrite",
        path=path,
        replacement="rewrite",
        family="aig_rewrite",
        metadata=(("command", "rewrite"),),
    )
    changed_path = AtomicEditOption(
        option_id="fixture.sequence_03.rewrite",
        path=JsonPath((ObjectKey("sequence"), ArrayIndex(4))),
        replacement="rewrite",
        family="aig_rewrite",
        metadata=(("command", "rewrite"),),
    )
    changed_replacement = AtomicEditOption(
        option_id="fixture.sequence_03.rewrite",
        path=path,
        replacement="rewrite_z",
        family="aig_rewrite",
        metadata=(("command", "rewrite"),),
    )
    changed_metadata = AtomicEditOption(
        option_id="fixture.sequence_03.rewrite",
        path=path,
        replacement="rewrite",
        family="aig_rewrite",
        metadata=(("command", "rewrite -z"),),
    )

    assert option == same
    assert option.identity_sha256 == same.identity_sha256
    assert len(
        {
            option.identity_sha256,
            changed_path.identity_sha256,
            changed_replacement.identity_sha256,
            changed_metadata.identity_sha256,
        }
    ) == 4
    with pytest.raises(FrozenInstanceError):
        option.family = "different"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {
                "option_id": "Uppercase",
                "path": JsonPath((ObjectKey("x"),)),
                "replacement": 1,
                "family": "family",
            },
            ValueError,
            "option_id",
        ),
        (
            {
                "option_id": "valid",
                "path": JsonPath(),
                "replacement": 1,
                "family": "family",
            },
            ValueError,
            "root",
        ),
        (
            {
                "option_id": "valid",
                "path": JsonPath((ObjectKey("x"),)),
                "replacement": freeze_json([1]),
                "family": "family",
            },
            TypeError,
            "scalar",
        ),
        (
            {
                "option_id": "valid",
                "path": JsonPath((ObjectKey("x"),)),
                "replacement": 1,
                "family": "family",
                "metadata": (("z", "last"), ("a", "first")),
            },
            ValueError,
            "canonically sorted",
        ),
        (
            {
                "option_id": "valid",
                "path": JsonPath((ObjectKey("x"),)),
                "replacement": 1,
                "family": "family",
                "metadata": (("a", "one"), ("a", "two")),
            },
            ValueError,
            "unique",
        ),
    ],
)
def test_atomic_edit_option_fails_closed(kwargs, error, match) -> None:
    with pytest.raises(error, match=match):
        AtomicEditOption(**kwargs)


def test_boils_catalog_is_protocol_conforming_deterministic_and_path_major() -> None:
    catalog = BoilsAtomicVariationCatalog()
    assert isinstance(catalog, AtomicVariationCatalog)
    first = catalog.options(_candidate())
    second = BoilsAtomicVariationCatalog().options(_candidate(suffix="copy"))

    assert first == second
    assert len(first) == SEQUENCE_LENGTH * (len(ACTION_IDS) - 1) == 200
    assert len({option.option_id for option in first}) == len(first)
    assert len({option.identity_sha256 for option in first}) == len(first)

    for index in range(SEQUENCE_LENGTH):
        coordinate = first[index * 10 : (index + 1) * 10]
        expected_path = JsonPath((ObjectKey("sequence"), ArrayIndex(index)))
        assert tuple(option.path for option in coordinate) == (expected_path,) * 10
        assert tuple(option.replacement for option in coordinate) == tuple(
            action
            for action in ACTION_IDS
            if action != DEFAULT_ACTION_SEQUENCE[index]
        )


def test_boils_options_bind_exact_commands_families_and_source_hashes() -> None:
    assert tuple(ACTION_FAMILIES) == ACTION_IDS
    assert tuple(ACTION_DEFINITION_SHA256) == ACTION_IDS
    assert CATALOG_SOURCE_SHA256 == (
        "86cec7f72595743acc5a1252e948ec7aa866dd268e72e500424f691e04b12d6a"
    )
    options = BoilsAtomicVariationCatalog().options(_candidate())

    for option in options:
        action_id = option.replacement
        assert type(action_id) is str
        assert option.family == ACTION_FAMILIES[action_id]
        metadata = dict(option.metadata)
        assert json.loads(metadata["abc_commands_json"]) == list(
            ACTION_COMMANDS[action_id]
        )
        assert metadata["action_definition_sha256"] == (
            ACTION_DEFINITION_SHA256[action_id]
        )
        assert metadata["catalog_source_sha256"] == CATALOG_SOURCE_SHA256
        binding = option.option_id.rsplit(".", 1)[-1]
        assert len(binding) == 64
        assert set(binding) <= set("0123456789abcdef")


def test_parent_relative_catalog_excludes_only_each_current_coordinate_value() -> None:
    original = BoilsAtomicVariationCatalog().options(_candidate())
    changed_sequence = list(DEFAULT_ACTION_SEQUENCE)
    changed_sequence[0] = "rewrite"
    changed = BoilsAtomicVariationCatalog().options(
        _candidate(tuple(changed_sequence), suffix="changed")
    )

    original_first = original[:10]
    changed_first = changed[:10]
    assert "balance" not in {option.replacement for option in original_first}
    assert "rewrite" not in {option.replacement for option in changed_first}
    assert "rewrite" in {option.replacement for option in original_first}
    assert "balance" in {option.replacement for option in changed_first}
    assert original[10:] == changed[10:]


def test_boils_catalog_revalidates_parent_schema_without_mutation() -> None:
    invalid = _candidate(("balance",), suffix="invalid")
    with pytest.raises(ValidationError):
        BoilsAtomicVariationCatalog().options(invalid)

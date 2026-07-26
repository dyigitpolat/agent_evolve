from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256
from agent_evolve.policies.selection.finite_option_prompt_projection import (
    FiniteOptionPromptProjectionPolicy,
    PromptMetadataProjectionMode,
)


def _contract(*, catalog_definition_sha256: str = "a" * 64) -> FiniteVariationContract:
    parent = freeze_json({"x": 0, "y": 0})
    parent_sha256 = typed_json_sha256(parent)
    options = tuple(
        FiniteVariationOption(
            option_id=f"synthetic.x{value}",
            parent_configuration_sha256=parent_sha256,
            child_configuration=freeze_json({"x": value, "y": 0}),
            family="coordinate_x",
            description=f"Set x from 0 to {value}.",
            metadata=(
                ("catalog_definition_sha256", catalog_definition_sha256),
                ("locus", "x"),
                ("target_value", str(value)),
            ),
        )
        for value in (1, 2, 3)
    )
    return FiniteVariationContract(
        catalog_id="synthetic_projection",
        catalog_version=1,
        catalog_definition_sha256=catalog_definition_sha256,
        parent_configuration=parent,
        options=options,
    )


def _encoded(records: tuple[dict[str, object], ...]) -> bytes:
    return json.dumps(
        list(records),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def test_all_mode_is_exactly_legacy_prompt_compatible() -> None:
    contract = _contract()
    projection = FiniteOptionPromptProjectionPolicy().project(contract)

    assert projection.mode is PromptMetadataProjectionMode.ALL
    assert projection.prompt_records() == contract.prompt_records()
    assert projection.omitted_metadata_keys == ()
    assert projection.to_binding_record()["outcome_values_consulted"] is False
    projection.require_contract(contract)


def test_allowlist_removes_only_explicit_metadata_and_authenticates_source() -> None:
    contract = _contract()
    policy = FiniteOptionPromptProjectionPolicy(metadata_keys=("locus", "target_value"))
    projection = policy.project(contract)

    assert projection.mode is PromptMetadataProjectionMode.ALLOWLIST
    assert projection.included_metadata_keys == ("locus", "target_value")
    assert projection.omitted_metadata_keys == ("catalog_definition_sha256",)
    assert all(
        tuple(record["metadata"]) == ("locus", "target_value")
        for record in projection.prompt_records()
    )
    assert len(_encoded(projection.prompt_records())) < len(
        _encoded(contract.prompt_records())
    )
    binding = projection.to_binding_record()
    assert binding["policy"]["configuration_sha256"] == (policy.configuration_sha256)
    assert len(binding["ordered_records"]) == len(contract.options)
    assert all(
        set(row)
        == {
            "option_id",
            "source_option_identity_sha256",
            "prompt_record_sha256",
        }
        for row in binding["ordered_records"]
    )
    projection.require_contract(contract)


def test_empty_allowlist_is_explicit_and_still_retains_action_semantics() -> None:
    contract = _contract()
    projection = FiniteOptionPromptProjectionPolicy(metadata_keys=()).project(contract)

    assert projection.mode is PromptMetadataProjectionMode.ALLOWLIST
    assert projection.included_metadata_keys == ()
    assert all(record["metadata"] == {} for record in projection.prompt_records())
    assert [record["description"] for record in projection.prompt_records()] == [
        option.description for option in contract.options
    ]


def test_unknown_or_noncanonical_allowlist_fails_closed() -> None:
    contract = _contract()

    with pytest.raises(ValueError, match="absent"):
        FiniteOptionPromptProjectionPolicy(metadata_keys=("unknown_key",)).project(
            contract
        )
    with pytest.raises(ValueError, match="canonical"):
        FiniteOptionPromptProjectionPolicy(metadata_keys=("target_value", "locus"))
    with pytest.raises(ValueError, match="canonical"):
        FiniteOptionPromptProjectionPolicy(metadata_keys=("locus", "locus"))


def test_projection_replay_rejects_foreign_contract_or_tampered_receipt() -> None:
    projection = FiniteOptionPromptProjectionPolicy(
        metadata_keys=("locus", "target_value")
    ).project(_contract())
    foreign = _contract(catalog_definition_sha256="b" * 64)

    with pytest.raises(ValueError, match="differs"):
        projection.require_contract(foreign)
    with pytest.raises(ValueError, match="configuration"):
        replace(projection, policy_configuration_sha256="0" * 64)
    with pytest.raises(ValueError, match="complement"):
        replace(projection, omitted_metadata_keys=())

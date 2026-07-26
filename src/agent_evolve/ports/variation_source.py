"""Workload-neutral source labels for sealed finite variation options.

Variation source is a property of how a candidate configuration entered the
finite action universe.  It is deliberately independent of whether a language
model retained the option or deterministic reconciliation inserted it.  The
latter remains semantic-provenance evidence; conflating the two would split one
proposal source across unrelated posterior arms.
"""

from __future__ import annotations

import re

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    FiniteVariationOption,
)


PRIMARY_VARIATION_SOURCE_ID = "primary"
VARIATION_SOURCE_METADATA_KEY = "evaluation_source"
VARIATION_SOURCE_MINIMUM_METADATA_KEY = "evaluation_source_minimum"
VARIATION_OPERATOR_METADATA_KEY = "evaluation_operator"
VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY = "evaluation_diversity_signature"
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")


def finite_variation_source_id(option: FiniteVariationOption) -> str:
    """Return one sealed source label, inheriting ``primary`` when absent."""

    if type(option) is not FiniteVariationOption:
        raise TypeError("option must be an exact FiniteVariationOption")
    option.__post_init__()
    source_id = dict(option.metadata).get(
        VARIATION_SOURCE_METADATA_KEY,
        PRIMARY_VARIATION_SOURCE_ID,
    )
    if _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("variation source ID must use the closed token grammar")
    return source_id


def finite_variation_source_by_option(
    contract: FiniteVariationContract,
) -> dict[str, str]:
    """Project a finite contract into exact option-to-source attribution."""

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    return {
        option.option_id: finite_variation_source_id(option)
        for option in contract.options
    }


def finite_variation_source_ids(
    contract: FiniteVariationContract,
) -> tuple[str, ...]:
    """Return the canonical source-arm universe declared by one contract."""

    return tuple(sorted(set(finite_variation_source_by_option(contract).values())))


def finite_variation_operator_id(option: FiniteVariationOption) -> str:
    """Return a typed operator arm without inspecting workload identifiers.

    New catalogs may declare ``evaluation_operator`` explicitly.  Existing
    authenticated radius-composition catalogs remain replayable through their
    generic ``composition_radius`` metadata; all other options inherit the
    primary atomic operator.
    """

    if type(option) is not FiniteVariationOption:
        raise TypeError("option must be an exact FiniteVariationOption")
    option.__post_init__()
    metadata = dict(option.metadata)
    operator_id = metadata.get(VARIATION_OPERATOR_METADATA_KEY)
    if operator_id is None:
        operator_id = "composite" if "composition_radius" in metadata else "atomic"
    if _SOURCE_ID.fullmatch(operator_id) is None:
        raise ValueError("variation operator ID must use the closed token grammar")
    return operator_id


def finite_variation_diversity_signature(option: FiniteVariationOption) -> str:
    """Return the adapter-declared typed diversity signature or action family.

    This is an optional inverted-API seam.  The default preserves the existing
    finite-family semantics; workloads can later expose a finer phenotype-safe
    signature without teaching the search controller domain names.
    """

    if type(option) is not FiniteVariationOption:
        raise TypeError("option must be an exact FiniteVariationOption")
    option.__post_init__()
    signature = dict(option.metadata).get(
        VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY,
        option.family,
    )
    if _SOURCE_ID.fullmatch(signature) is None:
        raise ValueError("diversity signature must use the closed token grammar")
    return signature


__all__ = [
    "PRIMARY_VARIATION_SOURCE_ID",
    "VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY",
    "VARIATION_OPERATOR_METADATA_KEY",
    "VARIATION_SOURCE_METADATA_KEY",
    "VARIATION_SOURCE_MINIMUM_METADATA_KEY",
    "finite_variation_diversity_signature",
    "finite_variation_operator_id",
    "finite_variation_source_by_option",
    "finite_variation_source_id",
    "finite_variation_source_ids",
]

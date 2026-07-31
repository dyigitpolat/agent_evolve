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
VARIATION_SOURCE_RANK_METADATA_KEY = "evaluation_source_rank"
VARIATION_OPERATOR_METADATA_KEY = "evaluation_operator"
VARIATION_DIVERSITY_SIGNATURE_METADATA_KEY = "evaluation_diversity_signature"
VARIATION_CANDIDATE_POOL_REQUIREMENT_METADATA_KEY = (
    "candidate_pool_membership_requirement"
)
_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,255}$")


def _source_id_from_validated_option(option: FiniteVariationOption) -> str:
    """Project source metadata after one enclosing graph validation."""

    source_id = dict(option.metadata).get(
        VARIATION_SOURCE_METADATA_KEY,
        PRIMARY_VARIATION_SOURCE_ID,
    )
    if _SOURCE_ID.fullmatch(source_id) is None:
        raise ValueError("variation source ID must use the closed token grammar")
    return source_id


def finite_variation_source_id(option: FiniteVariationOption) -> str:
    """Return one sealed source label, inheriting ``primary`` when absent."""

    if type(option) is not FiniteVariationOption:
        raise TypeError("option must be an exact FiniteVariationOption")
    option.__post_init__()
    return _source_id_from_validated_option(option)


def finite_variation_source_by_option(
    contract: FiniteVariationContract,
) -> dict[str, str]:
    """Project a finite contract into exact option-to-source attribution."""

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    return {
        option.option_id: _source_id_from_validated_option(option)
        for option in contract.options
    }


def finite_variation_source_ids(
    contract: FiniteVariationContract,
) -> tuple[str, ...]:
    """Return the canonical source-arm universe declared by one contract."""

    return tuple(sorted(set(finite_variation_source_by_option(contract).values())))


def finite_variation_source_minimum_counts(
    contract: FiniteVariationContract,
) -> tuple[tuple[str, int], ...]:
    """Project semantic evaluator-exposure floors from a finite contract.

    The result deliberately names source counts rather than representative
    option IDs.  Structural feasibility layers can therefore choose any
    compatible action from a source while retaining one workload-neutral
    inverted API.
    """

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    available_by_source: dict[str, int] = {}
    minimum_by_source: dict[str, int] = {}
    for option in contract.options:
        metadata = dict(option.metadata)
        source_id = metadata.get(VARIATION_SOURCE_METADATA_KEY)
        raw_minimum = metadata.get(VARIATION_SOURCE_MINIMUM_METADATA_KEY)
        if source_id is None and raw_minimum is None:
            continue
        if source_id is None:
            raise ValueError("source exposure minimum requires a source ID")
        if source_id != _source_id_from_validated_option(option):
            raise ValueError("source option metadata differs from its source ID")
        if raw_minimum is None:
            # Attribution and hard exposure are deliberately independent.
            continue
        if not raw_minimum.isascii() or not raw_minimum.isdigit():
            raise ValueError("evaluation source minimum must be decimal digits")
        minimum = int(raw_minimum)
        if not 1 <= minimum < 8:
            raise ValueError("evaluation source minimum must lie in [1, 8)")
        previous = minimum_by_source.setdefault(source_id, minimum)
        if previous != minimum:
            raise ValueError("one proposal source declares inconsistent minimums")
        available_by_source[source_id] = available_by_source.get(source_id, 0) + 1
    for source_id, minimum in minimum_by_source.items():
        if available_by_source[source_id] < minimum:
            raise ValueError("proposal source cannot satisfy its exposure minimum")
    return tuple(sorted(minimum_by_source.items()))


def finite_variation_candidate_pool_required_option_ids(
    contract: FiniteVariationContract,
) -> tuple[str, ...]:
    """Return options that must survive model-blind pool screening.

    Pool membership is intentionally distinct from evaluator allocation.  An
    upstream expert may need a complete reference slate exposed to the model
    and downstream allocator without forcing that slate to consume every real
    evaluation slot.  The closed metadata value prevents truthy-string or
    workload-specific interpretations at the common-pool boundary.
    """

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    required: list[str] = []
    for option in contract.options:
        raw = dict(option.metadata).get(
            VARIATION_CANDIDATE_POOL_REQUIREMENT_METADATA_KEY
        )
        if raw is None:
            continue
        if raw != "required":
            raise ValueError(
                "candidate-pool membership requirement must be exactly required"
            )
        required.append(option.option_id)
    return tuple(sorted(required))


def _operator_id_from_validated_option(option: FiniteVariationOption) -> str:
    """Project operator metadata after one enclosing graph validation."""

    metadata = dict(option.metadata)
    operator_id = metadata.get(VARIATION_OPERATOR_METADATA_KEY)
    if operator_id is None:
        operator_id = "composite" if "composition_radius" in metadata else "atomic"
    if _SOURCE_ID.fullmatch(operator_id) is None:
        raise ValueError("variation operator ID must use the closed token grammar")
    return operator_id


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
    return _operator_id_from_validated_option(option)


def finite_variation_operator_by_option(
    contract: FiniteVariationContract,
) -> dict[str, str]:
    """Project exact option-to-operator attribution in one validation pass."""

    if type(contract) is not FiniteVariationContract:
        raise TypeError("contract must be an exact FiniteVariationContract")
    contract.__post_init__()
    return {
        option.option_id: _operator_id_from_validated_option(option)
        for option in contract.options
    }


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
    "VARIATION_CANDIDATE_POOL_REQUIREMENT_METADATA_KEY",
    "VARIATION_OPERATOR_METADATA_KEY",
    "VARIATION_SOURCE_METADATA_KEY",
    "VARIATION_SOURCE_MINIMUM_METADATA_KEY",
    "VARIATION_SOURCE_RANK_METADATA_KEY",
    "finite_variation_diversity_signature",
    "finite_variation_candidate_pool_required_option_ids",
    "finite_variation_operator_by_option",
    "finite_variation_operator_id",
    "finite_variation_source_by_option",
    "finite_variation_source_id",
    "finite_variation_source_ids",
    "finite_variation_source_minimum_counts",
]

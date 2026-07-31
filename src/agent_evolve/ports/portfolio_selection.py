"""Framework-neutral contracts for one-call ranked action portfolios.

The selector sees immutable prompt views of versioned insight cards and one
parent-bound finite variation contract.  It returns an ordered portfolio of
opaque option IDs; trusted application code resolves every ID back to the
sealed option before a decision can be published.  The port deliberately does
not extend :class:`AgenticGenerator`: portfolio selection is an independently
replaceable policy, not another candidate-authoring mode.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from threading import RLock
from typing import Protocol, runtime_checkable

from agent_evolve.domain.finite_variation import (
    FiniteActionEvidenceBinding,
    FiniteVariationContract,
    ValidatedFiniteVariationIdentityIndex,
    validated_finite_variation_identity_index,
)
from agent_evolve.domain.ids import CandidateId, LLMCallId
from agent_evolve.domain.insight import InsightRef
from agent_evolve.domain.patch import JsonPath, require_sha256
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    freeze_json,
    thaw_json,
    typed_json_sha256,
)
from agent_evolve.ports.agentic_generator import (
    AgenticCallTelemetry,
    MetricEffectPrediction,
)
from agent_evolve.ports.portfolio_memory_dose import (
    BoundedPortfolioMemoryDoseContract,
    PortfolioMemoryDoseAssessment,
    PortfolioMemoryDoseMember,
    PortfolioMemoryDoseStage,
)
from agent_evolve.ports.structured_generator import MAX_OUTPUT_TOKENS
from agent_evolve.policies.variation.typed_patch import derive_patch


_TOKEN = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_OPTION_ID = re.compile(r"^[a-z][a-z0-9_.-]{0,255}$")
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_.:-]{0,191}$")
_REQUEST_DOMAIN = b"agent-evolve:portfolio-selection-request:v1\x00"
_CARD_SNAPSHOT_DOMAIN = b"agent-evolve:portfolio-card-snapshot:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:ranked-portfolio-decision:v1\x00"
_SUPPLEMENTAL_AUDIT_DOMAIN = (
    b"agent-evolve:portfolio-selection-supplemental-audit:v1\x00"
)
_TRANSFER_REQUEST_DOMAIN = b"agent-evolve:card-transfer-request:v1\x00"
_TRANSFER_RECEIPT_DOMAIN = b"agent-evolve:card-transfer-score-receipt:v1\x00"
_CARD_SCORE_STATE_DOMAIN = b"agent-evolve:portfolio-card-score-state:v1\x00"
_CARD_ACTION_EVIDENCE_STATE_DOMAIN = (
    b"agent-evolve:portfolio-card-action-evidence-state:v1\x00"
)
_CARD_SOURCE_BINDING_DOMAIN = b"agent-evolve:portfolio-card-source-binding:v1\x00"
_CARD_DERIVED_VIEW_DOMAIN = b"agent-evolve:portfolio-card-derived-view:v1\x00"
_CARD_SOURCE_REGISTRY_DOMAIN = b"agent-evolve:portfolio-card-source-registry:v1\x00"
_EXPERIMENTAL_VIEW_RECEIPT_DOMAIN = (
    b"agent-evolve:portfolio-experimental-view-receipt:v1\x00"
)
_REDACTED_EVIDENCE_DOMAIN = b"agent-evolve:portfolio-card-redacted-evidence:v1\x00"
_PARENT_PATCH_CERTIFICATE_DOMAIN = (
    b"agent-evolve:parent-patch-feasibility-certificate:v1\x00"
)
_PARENT_PATCH_WITNESS_ORDER_DOMAIN = (
    b"agent-evolve:parent-patch-feasibility-witness-order:v1\x00"
)
_MAX_PARENT_PATCH_CERTIFICATES = 64
_MAX_PARENT_PATCH_CACHED_PATHS = 65_536
_MAX_PARENT_PATCH_FEASIBILITY_RESULTS = 512


# Scientific neutral controls use one public, immutable representation.  An
# empty typed object contains no benchmark vocabulary and therefore cannot
# accidentally reveal an option through a nominally "neutral" sham card.
CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD = FrozenJsonObject(())
CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256 = hashlib.sha256(
    _REDACTED_EVIDENCE_DOMAIN
).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _reference_record(reference: InsightRef) -> dict[str, object]:
    if type(reference) is not InsightRef:
        raise TypeError("reference must be an exact InsightRef")
    InsightRef.__post_init__(reference)
    return {
        "insight_id": reference.insight_id.value,
        "version": reference.version,
    }


def _canonical_metric_ids(values: tuple[str, ...]) -> None:
    if type(values) is not tuple or any(
        type(value) is not str or _METRIC_ID.fullmatch(value) is None
        for value in values
    ):
        raise TypeError(
            "required_metric_ids must be an exact tuple of metric identifiers"
        )
    if not values:
        raise ValueError("required_metric_ids must be non-empty")
    if values != tuple(sorted(set(values))):
        raise ValueError("required_metric_ids must be unique and canonical")


@dataclass(frozen=True, slots=True)
class _ParentPatchFeasibilityCertificate:
    """One content-bound derivation of all parent-relative option paths."""

    contract_identity_sha256: str
    option_ids: tuple[str, ...]
    option_identity_sha256s: tuple[str, ...]
    option_families: tuple[str, ...]
    paths_by_option: tuple[tuple[JsonPath, ...], ...]
    certificate_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        require_sha256(self.contract_identity_sha256, "contract_identity_sha256")
        count = len(self.option_ids)
        if (
            count == 0
            or type(self.option_ids) is not tuple
            or any(
                type(value) is not str or _OPTION_ID.fullmatch(value) is None
                for value in self.option_ids
            )
            or len(set(self.option_ids)) != count
        ):
            raise ValueError("certificate option IDs must be distinct finite IDs")
        if (
            type(self.option_identity_sha256s) is not tuple
            or len(self.option_identity_sha256s) != count
        ):
            raise ValueError("certificate identities must cover every option")
        for value in self.option_identity_sha256s:
            require_sha256(value, "certificate option identity")
        if (
            type(self.option_families) is not tuple
            or len(self.option_families) != count
            or any(
                type(value) is not str or _TOKEN.fullmatch(value) is None
                for value in self.option_families
            )
        ):
            raise ValueError("certificate families must cover every option")
        if (
            type(self.paths_by_option) is not tuple
            or len(self.paths_by_option) != count
            or any(
                type(paths) is not tuple
                or not paths
                or any(type(path) is not JsonPath for path in paths)
                for paths in self.paths_by_option
            )
        ):
            raise ValueError("certificate paths must cover every option")
        object.__setattr__(
            self,
            "certificate_sha256",
            _hash(
                _PARENT_PATCH_CERTIFICATE_DOMAIN,
                {
                    "contract_identity_sha256": self.contract_identity_sha256,
                    "options": [
                        {
                            "option_id": option_id,
                            "option_identity_sha256": option_identity_sha256,
                            "family": family,
                            "path_schema_sha256s": [
                                path.schema_identity for path in paths
                            ],
                        }
                        for option_id, option_identity_sha256, family, paths in zip(
                            self.option_ids,
                            self.option_identity_sha256s,
                            self.option_families,
                            self.paths_by_option,
                            strict=True,
                        )
                    ],
                },
            ),
        )


_PARENT_PATCH_CACHE_LOCK = RLock()
_PARENT_PATCH_CERTIFICATES: OrderedDict[str, _ParentPatchFeasibilityCertificate] = (
    OrderedDict()
)
_PARENT_PATCH_CACHED_PATH_COUNT = 0
_PARENT_PATCH_FEASIBILITY_RESULTS: OrderedDict[
    tuple[str, tuple[str, ...], int, int | None], bool
] = OrderedDict()


def _parent_patch_certificate(
    contract: FiniteVariationContract,
) -> _ParentPatchFeasibilityCertificate:
    """Return a bounded-cache certificate after exact content validation."""

    index = validated_finite_variation_identity_index(contract)
    if type(index) is not ValidatedFiniteVariationIdentityIndex:
        raise TypeError("identity_index must be exact or None")
    index.__post_init__()
    key = index.contract_identity_sha256
    with _PARENT_PATCH_CACHE_LOCK:
        cached = _PARENT_PATCH_CERTIFICATES.get(key)
        if cached is not None:
            if (
                cached.option_ids != index.option_ids
                or cached.option_identity_sha256s != index.option_identity_sha256s
            ):
                raise RuntimeError("parent-patch certificate identity collision")
            _PARENT_PATCH_CERTIFICATES.move_to_end(key)
            return cached

    base_id = CandidateId("candidate_portfolio_disjoint_parent")
    target_id = CandidateId("candidate_portfolio_disjoint_child")
    paths: list[tuple[JsonPath, ...]] = []
    for option in contract.options:
        patch = derive_patch(
            contract.parent_configuration,
            option.child_configuration,
            base_candidate_id=base_id,
            target_candidate_id=target_id,
        )
        if not patch.operations:
            raise ValueError("finite option has no parent-relative patch")
        paths.append(tuple(operation.path for operation in patch.operations))
    certificate = _ParentPatchFeasibilityCertificate(
        contract_identity_sha256=key,
        option_ids=index.option_ids,
        option_identity_sha256s=index.option_identity_sha256s,
        option_families=tuple(option.family for option in contract.options),
        paths_by_option=tuple(paths),
    )
    certificate_path_count = sum(len(value) for value in certificate.paths_by_option)
    with _PARENT_PATCH_CACHE_LOCK:
        global _PARENT_PATCH_CACHED_PATH_COUNT
        prior = _PARENT_PATCH_CERTIFICATES.get(key)
        if prior is not None:
            if prior != certificate:
                raise RuntimeError("parent-patch certificate identity collision")
            _PARENT_PATCH_CERTIFICATES.move_to_end(key)
            return prior
        if certificate_path_count > _MAX_PARENT_PATCH_CACHED_PATHS:
            return certificate
        while _PARENT_PATCH_CERTIFICATES and (
            len(_PARENT_PATCH_CERTIFICATES) >= _MAX_PARENT_PATCH_CERTIFICATES
            or _PARENT_PATCH_CACHED_PATH_COUNT + certificate_path_count
            > _MAX_PARENT_PATCH_CACHED_PATHS
        ):
            _, evicted = _PARENT_PATCH_CERTIFICATES.popitem(last=False)
            _PARENT_PATCH_CACHED_PATH_COUNT -= sum(
                len(value) for value in evicted.paths_by_option
            )
        _PARENT_PATCH_CERTIFICATES[key] = certificate
        _PARENT_PATCH_CACHED_PATH_COUNT += certificate_path_count
    return certificate


def _parent_patch_paths_by_option(
    contract: FiniteVariationContract,
    option_ids: tuple[str, ...],
) -> dict[str, tuple[JsonPath, ...]]:
    """Resolve paths from one content-bound contract certificate."""

    if type(option_ids) is not tuple or any(
        type(value) is not str or _OPTION_ID.fullmatch(value) is None
        for value in option_ids
    ):
        raise TypeError("option_ids must be an exact tuple of finite option IDs")
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option_ids cannot repeat")
    certificate = _parent_patch_certificate(contract)
    all_paths = dict(
        zip(
            certificate.option_ids,
            certificate.paths_by_option,
            strict=True,
        )
    )
    if any(option_id not in all_paths for option_id in option_ids):
        raise ValueError("option_id is outside the sealed finite contract")
    return {option_id: all_paths[option_id] for option_id in option_ids}


def _path_sets_are_disjoint(
    left: tuple[JsonPath, ...],
    right: tuple[JsonPath, ...],
) -> bool:
    return not any(
        left_path.is_prefix_of(right_path) or right_path.is_prefix_of(left_path)
        for left_path in left
        for right_path in right
    )


def pairwise_disjoint_parent_patch_pairs(
    contract: FiniteVariationContract,
    option_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Return every canonical option pair with non-overlapping parent patches."""

    paths = _parent_patch_paths_by_option(contract, option_ids)
    ordered = tuple(sorted(option_ids))
    return tuple(
        (left, right)
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
        if _path_sets_are_disjoint(paths[left], paths[right])
    )


def _validate_family_exposure_bounds(
    values: tuple[tuple[str, int, int], ...],
    *,
    portfolio_size: int,
) -> None:
    if type(values) is not tuple:
        raise TypeError("family_exposure_bounds must be an exact tuple")
    if len(values) > portfolio_size:
        raise ValueError("family exposure bound count exceeds the portfolio size")
    families: list[str] = []
    minimum_total = 0
    for value in values:
        if type(value) is not tuple or len(value) != 3:
            raise TypeError("family exposure bounds must be exact three-tuples")
        family, minimum, maximum = value
        if type(family) is not str or not family:
            raise ValueError("family exposure family must be a non-empty string")
        if type(minimum) is not int or not 0 <= minimum <= portfolio_size:
            raise ValueError("family exposure minimum is outside the portfolio")
        if type(maximum) is not int or not 0 <= maximum <= portfolio_size:
            raise ValueError("family exposure maximum is outside the portfolio")
        if minimum > maximum:
            raise ValueError("family exposure minimum cannot exceed its maximum")
        families.append(family)
        minimum_total += minimum
    if tuple(families) != tuple(sorted(set(families))):
        raise ValueError("family exposure bounds must be unique and canonical")
    if minimum_total > portfolio_size:
        raise ValueError("family exposure minima exceed the portfolio size")


def _distinct_family_minimum_remains_reachable(
    *,
    chosen: tuple[str, ...],
    remaining: tuple[str, ...],
    portfolio_size: int,
    min_distinct_families: int | None,
    family_exposure_bounds: tuple[tuple[str, int, int], ...],
    family_by_id: dict[str, str],
) -> bool:
    """Return a cheap necessary condition for the distinct-family target.

    A family minimum greater than one consumes duplicate-family slots.  The
    former feasibility search ignored that interaction and could enumerate a
    large fraction of the K-subsets before proving, for example, that exact
    two members from one family is incompatible with four distinct families
    in a K=4 portfolio.  This bound accounts for mandatory remaining slots and
    then gives every free slot the optimistic benefit of introducing a new
    family.  It is therefore safe pruning: ``False`` proves infeasibility,
    while ``True`` leaves the exact patch-disjoint search unchanged.
    """

    if min_distinct_families is None:
        return True
    needed = portfolio_size - len(chosen)
    chosen_counts: dict[str, int] = {}
    for option_id in chosen:
        family = family_by_id[option_id]
        chosen_counts[family] = chosen_counts.get(family, 0) + 1
    remaining_counts: dict[str, int] = {}
    for option_id in remaining:
        family = family_by_id[option_id]
        remaining_counts[family] = remaining_counts.get(family, 0) + 1

    maximum_by_family = {
        family: maximum for family, _, maximum in family_exposure_bounds
    }
    mandatory_slots = 0
    mandatory_new_families: set[str] = set()
    for family, minimum, _ in family_exposure_bounds:
        outstanding = max(minimum - chosen_counts.get(family, 0), 0)
        if outstanding > remaining_counts.get(family, 0):
            return False
        mandatory_slots += outstanding
        if outstanding and family not in chosen_counts:
            mandatory_new_families.add(family)
    if mandatory_slots > needed:
        return False

    optional_new_families = {
        family
        for family in remaining_counts
        if family not in chosen_counts
        and family not in mandatory_new_families
        and maximum_by_family.get(family, portfolio_size) > 0
    }
    optimistic_distinct = (
        len(chosen_counts)
        + len(mandatory_new_families)
        + min(needed - mandatory_slots, len(optional_new_families))
    )
    return optimistic_distinct >= min_distinct_families


def finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
    contract: FiniteVariationContract,
    option_ids: tuple[str, ...],
    *,
    portfolio_size: int,
    min_distinct_families: int | None = None,
    family_exposure_bounds: tuple[tuple[str, int, int], ...] = (),
) -> bool:
    """Decide exact K-subset feasibility with branch pruning and early exit.

    Exact K-clique feasibility is exponential in the worst case.  The public
    finite contract is bounded to 1,024 options and portfolio sizes are small;
    this search stops on its first witness and prunes every undersized suffix.
    """

    if type(portfolio_size) is not int or portfolio_size <= 0:
        raise ValueError("portfolio_size must be a positive exact integer")
    if min_distinct_families is not None and (
        type(min_distinct_families) is not int
        or not 1 <= min_distinct_families <= portfolio_size
    ):
        raise ValueError(
            "min_distinct_families must lie within the requested portfolio"
        )
    _validate_family_exposure_bounds(
        family_exposure_bounds,
        portfolio_size=portfolio_size,
    )
    if type(option_ids) is not tuple or any(
        type(value) is not str or _OPTION_ID.fullmatch(value) is None
        for value in option_ids
    ):
        raise TypeError("option_ids must be an exact tuple of finite option IDs")
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option_ids cannot repeat")
    if portfolio_size > len(option_ids):
        return False
    certificate = _parent_patch_certificate(contract)
    path_by_id = dict(
        zip(certificate.option_ids, certificate.paths_by_option, strict=True)
    )
    family_by_id = dict(
        zip(certificate.option_ids, certificate.option_families, strict=True)
    )
    if any(option_id not in path_by_id for option_id in option_ids):
        raise ValueError("option_id is outside the sealed finite contract")
    ordered_option_ids = tuple(sorted(option_ids))
    cache_key = (
        certificate.certificate_sha256,
        ordered_option_ids,
        portfolio_size,
        min_distinct_families,
        family_exposure_bounds,
    )
    with _PARENT_PATCH_CACHE_LOCK:
        cached = _PARENT_PATCH_FEASIBILITY_RESULTS.get(cache_key)
        if cached is not None:
            _PARENT_PATCH_FEASIBILITY_RESULTS.move_to_end(cache_key)
            return cached

    def search(remaining: tuple[str, ...], chosen: tuple[str, ...]) -> bool:
        needed = portfolio_size - len(chosen)
        if needed == 0:
            if (
                min_distinct_families is not None
                and len({family_by_id[value] for value in chosen})
                < min_distinct_families
            ):
                return False
            return all(
                minimum
                <= sum(family_by_id[value] == family for value in chosen)
                <= maximum
                for family, minimum, maximum in family_exposure_bounds
            )
        if len(remaining) < needed:
            return False
        for family, minimum, maximum in family_exposure_bounds:
            chosen_count = sum(family_by_id[value] == family for value in chosen)
            if chosen_count > maximum:
                return False
            remaining_count = sum(
                family_by_id[value] == family for value in remaining
            )
            if chosen_count + min(needed, remaining_count) < minimum:
                return False
        if not _distinct_family_minimum_remains_reachable(
            chosen=chosen,
            remaining=remaining,
            portfolio_size=portfolio_size,
            min_distinct_families=min_distinct_families,
            family_exposure_bounds=family_exposure_bounds,
            family_by_id=family_by_id,
        ):
            return False
        for index, option_id in enumerate(remaining):
            if any(
                family_by_id[option_id] == family
                and sum(family_by_id[value] == family for value in chosen)
                >= maximum
                for family, _, maximum in family_exposure_bounds
            ):
                continue
            if all(
                _path_sets_are_disjoint(path_by_id[option_id], path_by_id[prior])
                for prior in chosen
            ) and search(remaining[index + 1 :], (*chosen, option_id)):
                return True
        return False

    result = search(ordered_option_ids, ())
    with _PARENT_PATCH_CACHE_LOCK:
        _PARENT_PATCH_FEASIBILITY_RESULTS[cache_key] = result
        _PARENT_PATCH_FEASIBILITY_RESULTS.move_to_end(cache_key)
        while (
            len(_PARENT_PATCH_FEASIBILITY_RESULTS)
            > _MAX_PARENT_PATCH_FEASIBILITY_RESULTS
        ):
            _PARENT_PATCH_FEASIBILITY_RESULTS.popitem(last=False)
    return result


def pairwise_disjoint_parent_patch_witness(
    contract: FiniteVariationContract,
    option_ids: tuple[str, ...],
    *,
    portfolio_size: int,
    min_distinct_families: int | None = None,
    family_exposure_bounds: tuple[tuple[str, int, int], ...] = (),
    ordering_key_sha256: str | None = None,
    preferred_option_ids: tuple[str, ...] = (),
    required_option_ids: tuple[str, ...] = (),
) -> tuple[str, ...] | None:
    """Return one deterministic feasibility witness, or ``None`` if absent.

    The witness is derived only from the sealed parent-relative patches and
    option families.  It is therefore safe to expose before evaluation as a
    structural decoding aid: it proves how to satisfy the hard combinatorial
    contract without ranking an option by objective quality.  The default
    preserves canonical option-ID order.  An optional authenticated SHA-256
    key gives independent requests deterministic, domain-separated orderings
    without consulting outcomes, option prose, or workload semantics.  When
    ``required_option_ids`` is a hard membership constraint with no ranking
    semantics.  When ``preferred_option_ids`` is non-empty, the witness first
    includes every required option, then maximizes the number of retained
    preferred options, and finally prefers the lexicographically earliest
    preferred-rank combination. Canonical/keyed order breaks ties only among
    nonpreferred completions. This supports an outcome-blind protected-source
    floor alongside minimum-intervention semantic projection.
    """

    if type(portfolio_size) is not int or portfolio_size <= 0:
        raise ValueError("portfolio_size must be a positive exact integer")
    if min_distinct_families is not None and (
        type(min_distinct_families) is not int
        or not 1 <= min_distinct_families <= portfolio_size
    ):
        raise ValueError(
            "min_distinct_families must lie within the requested portfolio"
        )
    _validate_family_exposure_bounds(
        family_exposure_bounds,
        portfolio_size=portfolio_size,
    )
    if type(option_ids) is not tuple or any(
        type(value) is not str or _OPTION_ID.fullmatch(value) is None
        for value in option_ids
    ):
        raise TypeError("option_ids must be an exact tuple of finite option IDs")
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("option_ids cannot repeat")
    if type(preferred_option_ids) is not tuple or any(
        type(value) is not str or _OPTION_ID.fullmatch(value) is None
        for value in preferred_option_ids
    ):
        raise TypeError(
            "preferred_option_ids must be an exact tuple of finite option IDs"
        )
    if len(set(preferred_option_ids)) != len(preferred_option_ids):
        raise ValueError("preferred_option_ids cannot repeat")
    if not set(preferred_option_ids).issubset(option_ids):
        raise ValueError("preferred_option_ids must be drawn from option_ids")
    if type(required_option_ids) is not tuple or any(
        type(value) is not str or _OPTION_ID.fullmatch(value) is None
        for value in required_option_ids
    ):
        raise TypeError(
            "required_option_ids must be an exact tuple of finite option IDs"
        )
    if required_option_ids != tuple(sorted(set(required_option_ids))):
        raise ValueError("required_option_ids must be unique and canonical")
    if not set(required_option_ids).issubset(option_ids):
        raise ValueError("required_option_ids must be drawn from option_ids")
    if len(required_option_ids) > portfolio_size:
        return None
    if portfolio_size > len(option_ids):
        return None
    if ordering_key_sha256 is not None:
        require_sha256(ordering_key_sha256, "ordering_key_sha256")

    certificate = _parent_patch_certificate(contract)
    path_by_id = dict(
        zip(certificate.option_ids, certificate.paths_by_option, strict=True)
    )
    family_by_id = dict(
        zip(certificate.option_ids, certificate.option_families, strict=True)
    )
    if any(option_id not in path_by_id for option_id in option_ids):
        raise ValueError("option_id is outside the sealed finite contract")
    if ordering_key_sha256 is None:
        ordered_option_ids = tuple(sorted(option_ids))
    else:
        ordering_key = bytes.fromhex(ordering_key_sha256)
        ordered_option_ids = tuple(
            sorted(
                option_ids,
                key=lambda option_id: (
                    hashlib.sha256(
                        _PARENT_PATCH_WITNESS_ORDER_DOMAIN
                        + ordering_key
                        + option_id.encode("ascii", errors="strict")
                    ).digest(),
                    option_id,
                ),
            )
        )

    def search(
        remaining: tuple[str, ...],
        chosen: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        needed = portfolio_size - len(chosen)
        if needed == 0:
            if (
                min_distinct_families is not None
                and len({family_by_id[value] for value in chosen})
                < min_distinct_families
            ):
                return None
            if any(
                not minimum
                <= sum(family_by_id[value] == family for value in chosen)
                <= maximum
                for family, minimum, maximum in family_exposure_bounds
            ):
                return None
            return chosen
        if len(remaining) < needed:
            return None
        for family, minimum, maximum in family_exposure_bounds:
            chosen_count = sum(family_by_id[value] == family for value in chosen)
            if chosen_count > maximum:
                return None
            remaining_count = sum(
                family_by_id[value] == family for value in remaining
            )
            if chosen_count + min(needed, remaining_count) < minimum:
                return None
        if not _distinct_family_minimum_remains_reachable(
            chosen=chosen,
            remaining=remaining,
            portfolio_size=portfolio_size,
            min_distinct_families=min_distinct_families,
            family_exposure_bounds=family_exposure_bounds,
            family_by_id=family_by_id,
        ):
            return None
        for index, option_id in enumerate(remaining):
            if any(
                family_by_id[option_id] == family
                and sum(family_by_id[value] == family for value in chosen)
                >= maximum
                for family, _, maximum in family_exposure_bounds
            ):
                continue
            if not all(
                _path_sets_are_disjoint(path_by_id[option_id], path_by_id[prior])
                for prior in chosen
            ):
                continue
            witness = search(remaining[index + 1 :], (*chosen, option_id))
            if witness is not None:
                return witness
        return None

    required_set = set(required_option_ids)
    if any(
        not _path_sets_are_disjoint(path_by_id[left], path_by_id[right])
        for index, left in enumerate(required_option_ids)
        for right in required_option_ids[index + 1 :]
    ):
        return None
    if any(
        sum(family_by_id[value] == family for value in required_option_ids)
        > maximum
        for family, _, maximum in family_exposure_bounds
    ):
        return None
    completion_option_ids = tuple(
        value for value in ordered_option_ids if value not in required_set
    )
    remaining_preferred = tuple(
        value for value in preferred_option_ids if value not in required_set
    )
    if not remaining_preferred:
        return search(completion_option_ids, required_option_ids)

    preferred_set = set(remaining_preferred)
    nonpreferred_option_ids = tuple(
        value for value in completion_option_ids if value not in preferred_set
    )
    maximum_retained = min(
        portfolio_size - len(required_option_ids),
        len(remaining_preferred),
    )
    for retained_count in range(maximum_retained, -1, -1):
        for preferred_subset in combinations(
            remaining_preferred,
            retained_count,
        ):
            if any(
                not _path_sets_are_disjoint(
                    path_by_id[left],
                    path_by_id[right],
                )
                for index, left in enumerate(preferred_subset)
                for right in preferred_subset[index + 1 :]
            ):
                continue
            if any(
                not _path_sets_are_disjoint(
                    path_by_id[required],
                    path_by_id[preferred],
                )
                for required in required_option_ids
                for preferred in preferred_subset
            ):
                continue
            if any(
                sum(
                    family_by_id[value] == family
                    for value in (*required_option_ids, *preferred_subset)
                )
                > maximum
                for family, _, maximum in family_exposure_bounds
            ):
                continue
            witness = search(
                nonpreferred_option_ids,
                (*required_option_ids, *preferred_subset),
            )
            if witness is not None:
                return witness
    return None


def project_family_exposure_bounds_to_pairwise_disjoint_feasibility(
    contract: FiniteVariationContract,
    option_ids: tuple[str, ...],
    *,
    portfolio_size: int,
    min_distinct_families: int | None = None,
    requested_bounds: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[str, int, int], ...]:
    """Project requested family bounds to the closest structural K-feasibility.

    The projection is outcome-blind. It first preserves the requested intervals
    exactly when any pairwise-disjoint K-subset satisfies them. Otherwise it
    chooses the feasible exact count vector with minimum L1 interval violation,
    using canonical count order only to break ties.
    """

    _validate_family_exposure_bounds(
        requested_bounds,
        portfolio_size=portfolio_size,
    )
    if not requested_bounds:
        return ()
    if finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
        contract,
        option_ids,
        portfolio_size=portfolio_size,
        min_distinct_families=min_distinct_families,
        family_exposure_bounds=requested_bounds,
    ):
        return requested_bounds
    count_vectors = tuple(
        sorted(
            product(range(portfolio_size + 1), repeat=len(requested_bounds)),
            key=lambda counts: (
                sum(
                    max(minimum - count, 0, count - maximum)
                    for (_, minimum, maximum), count in zip(
                        requested_bounds,
                        counts,
                        strict=True,
                    )
                ),
                counts,
            ),
        )
    )
    for counts in count_vectors:
        # Action families are mutually exclusive labels on each selected
        # option.  Exact counts whose sum exceeds K cannot describe any
        # portfolio and are outside the validated bound domain.
        if sum(counts) > portfolio_size:
            continue
        exact = tuple(
            (family, count, count)
            for (family, _, _), count in zip(
                requested_bounds,
                counts,
                strict=True,
            )
        )
        if not finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
            contract,
            option_ids,
            portfolio_size=portfolio_size,
            min_distinct_families=min_distinct_families,
            family_exposure_bounds=exact,
        ):
            continue
        return exact
    raise ValueError("finite option universe has no feasible evaluation portfolio")


def finite_portfolio_has_pairwise_disjoint_parent_patches(
    contract: FiniteVariationContract,
    *,
    portfolio_size: int,
    min_distinct_families: int | None = None,
) -> bool:
    """Decide whether a finite contract contains one disjoint K-portfolio."""

    return finite_option_ids_have_pairwise_disjoint_parent_patch_subset(
        contract,
        tuple(option.option_id for option in contract.options),
        portfolio_size=portfolio_size,
        min_distinct_families=min_distinct_families,
    )


def validate_pairwise_disjoint_parent_patch_selection(
    contract: FiniteVariationContract,
    option_ids: tuple[str, ...],
) -> None:
    """Fail closed unless every selected sealed option has a disjoint patch."""

    paths = _parent_patch_paths_by_option(contract, option_ids)
    for left_index, left in enumerate(option_ids):
        for right in option_ids[left_index + 1 :]:
            if not _path_sets_are_disjoint(paths[left], paths[right]):
                raise ValueError(
                    "selected finite options have overlapping parent-relative patches"
                )


@dataclass(frozen=True, slots=True)
class CardScoreComponent:
    """One interpretable scalar in a card's selector-visible score bundle.

    Components retain their scoring definition and evidence lineage so a
    benchmark can rotate, redact, or independently adjudicate complete score
    bundles without collapsing distinct semantics into an opaque global
    score.  ``evidence_count`` may exceed the number of receipts because one
    sealed receipt can summarize multiple evidence rows.
    """

    score_id: str
    value: float
    definition_sha256: str
    evidence_count: int
    receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.score_id) is not str
            or _METRIC_ID.fullmatch(self.score_id) is None
        ):
            raise ValueError("score_id must use the closed metric-token grammar")
        if type(self.value) is not float or not math.isfinite(self.value):
            raise TypeError("value must be a finite canonical float")
        require_sha256(self.definition_sha256, "definition_sha256")
        if type(self.evidence_count) is not int or self.evidence_count < 0:
            raise ValueError("evidence_count must be a non-negative exact integer")
        if type(self.receipt_sha256s) is not tuple or any(
            type(value) is not str for value in self.receipt_sha256s
        ):
            raise TypeError("receipt_sha256s must be an exact tuple of strings")
        for value in self.receipt_sha256s:
            require_sha256(value, "receipt_sha256")
        if self.receipt_sha256s != tuple(sorted(set(self.receipt_sha256s))):
            raise ValueError("receipt_sha256s must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "score_id": self.score_id,
            "value_hex": self.value.hex(),
            "definition_sha256": self.definition_sha256,
            "evidence_count": self.evidence_count,
            "receipt_sha256s": list(self.receipt_sha256s),
        }

    def prompt_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "score_id": self.score_id,
            "value": self.value,
            "definition_sha256": self.definition_sha256,
            "evidence_count": self.evidence_count,
            "receipt_sha256s": list(self.receipt_sha256s),
        }


def _validate_finite_action_evidence(
    values: tuple[FiniteActionEvidenceBinding, ...],
    *,
    name: str,
) -> None:
    if type(values) is not tuple or any(
        type(binding) is not FiniteActionEvidenceBinding for binding in values
    ):
        raise TypeError(
            f"{name} must be an exact tuple of FiniteActionEvidenceBinding values"
        )
    for binding in values:
        FiniteActionEvidenceBinding.__post_init__(binding)
    contrast_ids = tuple(binding.contrast_id for binding in values)
    if contrast_ids != tuple(sorted(set(contrast_ids))):
        raise ValueError(f"{name} must have unique canonical contrast order")


@dataclass(frozen=True, slots=True)
class PortfolioCardPromptPayload:
    """Typed scientific prompt boundary for one portfolio card.

    Free-form benchmark prose is confined to ``action_neutral_payload``.
    Exact option attribution is carried only by the structured finite-action
    bindings.  This type does not itself know the request's complete option
    catalog; :func:`validate_portfolio_experimental_view` performs that
    catalog-relative leak check before a scientific request is admitted.
    """

    action_neutral_payload: FrozenJsonObject
    finite_action_evidence: tuple[FiniteActionEvidenceBinding, ...]

    def __post_init__(self) -> None:
        if type(self.action_neutral_payload) is not FrozenJsonObject:
            raise TypeError("action_neutral_payload must be an exact FrozenJsonObject")
        if freeze_json(self.action_neutral_payload) is not self.action_neutral_payload:
            raise TypeError("action_neutral_payload must already be frozen typed JSON")
        _validate_finite_action_evidence(
            self.finite_action_evidence,
            name="finite_action_evidence",
        )

    @property
    def action_neutral_payload_sha256(self) -> str:
        self.__post_init__()
        return typed_json_sha256(self.action_neutral_payload)

    def prompt_record(self) -> dict[str, object]:
        """Render action-neutral prose and exact attribution separately."""

        self.__post_init__()
        return {
            "action_neutral_payload": thaw_json(self.action_neutral_payload),
            "finite_action_evidence": [
                binding.to_record() for binding in self.finite_action_evidence
            ],
        }

    def to_record(self) -> dict[str, object]:
        """Return a canonical evidence record without duplicating neutral prose."""

        self.__post_init__()
        return {
            "schema_version": 1,
            "action_neutral_payload_sha256": self.action_neutral_payload_sha256,
            "finite_action_evidence": [
                binding.to_record() for binding in self.finite_action_evidence
            ],
        }


def _typed_json_text_values(
    value: FrozenJsonValue,
) -> tuple[tuple[str, str], ...]:
    """Return paths and every key/string in an already-frozen JSON tree."""

    found: list[tuple[str, str]] = []

    def visit(node: FrozenJsonValue, path: str) -> None:
        if type(node) is str:
            found.append((path, node))
            return
        if type(node) is FrozenJsonArray:
            for index, item in enumerate(node.items):
                visit(item, f"{path}[{index}]")
            return
        if type(node) is FrozenJsonObject:
            for key, item in node.items:
                found.append((f"{path}.<key>", key))
                visit(item, f"{path}.{key}")

    visit(value, "$")
    return tuple(found)


def _contains_explicit_option_id(text: str, option_id: str) -> bool:
    """Detect catalog option identity anywhere in supposedly neutral text.

    A token-boundary check is unsafe because ``.`` is both legal inside an
    option ID and ordinary sentence punctuation: ``"choose alpha.x1."`` would
    otherwise evade the guard.  Scientific neutral payloads have no legitimate
    reason to contain a catalog option ID even as a substring, so the stricter
    containment rule is intentional.
    """

    return option_id in text.casefold()


def _validate_action_neutral_payload(
    payload: PortfolioCardPromptPayload,
    *,
    identity_index: ValidatedFiniteVariationIdentityIndex,
) -> None:
    """Fail closed when neutral prose carries exact finite-option identity."""

    if type(payload) is not PortfolioCardPromptPayload:
        raise TypeError("payload must be an exact PortfolioCardPromptPayload")
    payload.__post_init__()
    if type(identity_index) is not ValidatedFiniteVariationIdentityIndex:
        raise TypeError("identity_index must be an exact validated identity index")
    identity_index.__post_init__()

    source_option_ids = {
        binding.option_id for binding in payload.finite_action_evidence
    }
    forbidden_hashes = {
        identity_index.contract_identity_sha256,
        *identity_index.option_identity_sha256s,
        *identity_index.child_configuration_sha256s,
        *(binding.contrast_id for binding in payload.finite_action_evidence),
        *(
            binding.option_identity_sha256
            for binding in payload.finite_action_evidence
        ),
        *(
            binding.contract_identity_sha256
            for binding in payload.finite_action_evidence
        ),
        *(binding.identity_sha256 for binding in payload.finite_action_evidence),
    }
    for path, text in _typed_json_text_values(payload.action_neutral_payload):
        folded = text.casefold()
        for option_id in identity_index.option_ids:
            if _contains_explicit_option_id(folded, option_id):
                raise ValueError(
                    "scientific action-neutral payload contains explicit "
                    f"option_id at {path}"
                )
        for option_id in source_option_ids:
            if _contains_explicit_option_id(folded, option_id):
                raise ValueError(
                    "scientific action-neutral payload contains source "
                    f"option_id at {path}"
                )
        if any(value in folded for value in forbidden_hashes):
            raise ValueError(
                "scientific action-neutral payload contains exact option or "
                f"contract attribution at {path}"
            )


def portfolio_card_action_evidence_sha256(
    values: tuple[FiniteActionEvidenceBinding, ...],
) -> str:
    """Hash-bind the exact prompt-visible action-attribution view."""

    _validate_finite_action_evidence(values, name="finite_action_evidence")
    return _hash(
        _CARD_ACTION_EVIDENCE_STATE_DOMAIN,
        {
            "schema_version": 1,
            "finite_action_evidence": [value.to_record() for value in values],
        },
    )


def _card_score_state_record(
    score_components: tuple[CardScoreComponent, ...],
    assigned_score: float | None,
) -> dict[str, object]:
    if type(score_components) is not tuple or any(
        type(component) is not CardScoreComponent for component in score_components
    ):
        raise TypeError(
            "score_components must be an exact tuple of CardScoreComponent values"
        )
    for component in score_components:
        component.__post_init__()
    score_ids = tuple(component.score_id for component in score_components)
    if score_ids != tuple(sorted(set(score_ids))):
        raise ValueError("score_components must use unique canonical score_id order")
    if assigned_score is not None and (
        type(assigned_score) is not float or not math.isfinite(assigned_score)
    ):
        raise TypeError("assigned_score must be a finite canonical float or None")
    return {
        "score_components": [component.to_record() for component in score_components],
        "assigned_score_hex": (
            None if assigned_score is None else assigned_score.hex()
        ),
    }


def portfolio_card_score_state_sha256(
    score_components: tuple[CardScoreComponent, ...],
    assigned_score: float | None,
) -> str:
    """Hash-bind the exact score projection carried by one card view."""

    return _hash(
        _CARD_SCORE_STATE_DOMAIN,
        _card_score_state_record(score_components, assigned_score),
    )


@dataclass(frozen=True, slots=True)
class PortfolioCardSourceBinding:
    """Integrity-bound source provenance for one insight-to-card projection.

    This receipt binds the immutable insight version and evidence lineage to the
    exact source view constructed by trusted application code.  A later
    experimental view may redact or permute that view, but it must retain this
    receipt unchanged and publish a separate :class:`PortfolioCardViewReceipt`.
    """

    reference: InsightRef
    content_sha256: str
    evidence_lineage_identity_sha256: str
    finite_action_evidence: tuple[FiniteActionEvidenceBinding, ...]
    source_prompt_view_sha256: str
    source_evidence_sha256: str
    source_score_state_sha256: str
    source_receipt_sha256: str

    def __post_init__(self) -> None:
        _reference_record(self.reference)
        for name in (
            "content_sha256",
            "evidence_lineage_identity_sha256",
            "source_prompt_view_sha256",
            "source_evidence_sha256",
            "source_score_state_sha256",
            "source_receipt_sha256",
        ):
            require_sha256(getattr(self, name), name)
        _validate_finite_action_evidence(
            self.finite_action_evidence,
            name="finite_action_evidence",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "reference": _reference_record(self.reference),
            "content_sha256": self.content_sha256,
            "evidence_lineage_identity_sha256": (self.evidence_lineage_identity_sha256),
            "finite_action_evidence": [
                binding.to_record() for binding in self.finite_action_evidence
            ],
            "source_prompt_view_sha256": self.source_prompt_view_sha256,
            "source_evidence_sha256": self.source_evidence_sha256,
            "source_score_state_sha256": self.source_score_state_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
        }

    @property
    def binding_sha256(self) -> str:
        return _hash(_CARD_SOURCE_BINDING_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "binding_sha256": self.binding_sha256}


def _bind_portfolio_card_source(
    *,
    reference: InsightRef,
    content_sha256: str,
    evidence_lineage_identity_sha256: str,
    finite_action_evidence: tuple[FiniteActionEvidenceBinding, ...],
    prompt_payload: FrozenJsonObject,
    evidence_sha256: str,
    score_components: tuple[CardScoreComponent, ...],
    assigned_score: float | None,
    source_receipt_sha256: str,
) -> PortfolioCardSourceBinding:
    """Issue a source binding from exact framework-neutral projection values."""

    if type(prompt_payload) is not FrozenJsonObject:
        raise TypeError("prompt_payload must be an exact FrozenJsonObject")
    if freeze_json(prompt_payload) is not prompt_payload:
        raise TypeError("prompt_payload must already be frozen typed JSON")
    return PortfolioCardSourceBinding(
        reference=reference,
        content_sha256=content_sha256,
        evidence_lineage_identity_sha256=evidence_lineage_identity_sha256,
        finite_action_evidence=finite_action_evidence,
        source_prompt_view_sha256=typed_json_sha256(prompt_payload),
        source_evidence_sha256=evidence_sha256,
        source_score_state_sha256=portfolio_card_score_state_sha256(
            score_components,
            assigned_score,
        ),
        source_receipt_sha256=source_receipt_sha256,
    )


@dataclass(frozen=True, slots=True, init=False)
class PortfolioCardSourceRegistry:
    """Application-admitted source bindings for one scientific request.

    The registry is a hash-bound integrity and admission receipt, not a
    cryptographic signature.  Its public constructor is closed: trusted
    application code issues it only after joining every binding to an exact
    supplied memory entry.  Request validation then prevents a source binding
    synthesized directly by an adapter from entering without that admission.
    """

    source_bindings: tuple[PortfolioCardSourceBinding, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError(
            "PortfolioCardSourceRegistry is issued by trusted application code"
        )

    def __post_init__(self) -> None:
        if type(self.source_bindings) is not tuple or not self.source_bindings:
            raise ValueError("source_bindings must be a non-empty exact tuple")
        if any(
            type(binding) is not PortfolioCardSourceBinding
            for binding in self.source_bindings
        ):
            raise TypeError(
                "source_bindings must contain exact PortfolioCardSourceBinding values"
            )
        for binding in self.source_bindings:
            binding.__post_init__()
        identities = tuple(binding.binding_sha256 for binding in self.source_bindings)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("source_bindings must use unique canonical binding order")
        references = tuple(binding.reference for binding in self.source_bindings)
        if len(set(references)) != len(references):
            raise ValueError("source_bindings cannot repeat an insight reference")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_bindings": [
                binding.to_record() for binding in self.source_bindings
            ],
        }

    @property
    def registry_sha256(self) -> str:
        return _hash(_CARD_SOURCE_REGISTRY_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "registry_sha256": self.registry_sha256,
        }


def _issue_portfolio_card_source_registry(
    source_bindings: tuple[PortfolioCardSourceBinding, ...],
) -> PortfolioCardSourceRegistry:
    """Internal issuer used only after the application validates source joins."""

    registry = object.__new__(PortfolioCardSourceRegistry)
    object.__setattr__(registry, "source_bindings", source_bindings)
    registry.__post_init__()
    return registry


class PortfolioCardViewTransform(str, Enum):
    """Closed declarations for an experiment-controlled derived card view."""

    EVIDENCE_PERMUTATION = "evidence_permutation"
    EVIDENCE_REDACTION = "evidence_redaction"
    PROMPT_PERMUTATION = "prompt_permutation"
    PROMPT_PROJECTION = "prompt_projection"
    PROMPT_REDACTION = "prompt_redaction"
    SCORE_PERMUTATION = "score_permutation"
    SCORE_REDACTION = "score_redaction"
    ACTION_EVIDENCE_PERMUTATION = "action_evidence_permutation"
    ACTION_EVIDENCE_REDACTION = "action_evidence_redaction"


@dataclass(frozen=True, slots=True)
class PortfolioCardViewReceipt:
    """Receipt for a derived view that never replaces its source provenance."""

    source_binding_sha256: str
    transforms: tuple[PortfolioCardViewTransform, ...]
    derived_prompt_view_sha256: str
    derived_evidence_sha256: str
    derived_score_state_sha256: str
    derived_action_evidence_sha256: str
    prompt_source_binding_sha256: str | None
    evidence_source_binding_sha256: str | None
    score_source_binding_sha256: str | None
    action_evidence_source_binding_sha256: str | None
    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.source_binding_sha256, "source_binding_sha256")
        if (
            type(self.transforms) is not tuple
            or not self.transforms
            or any(
                type(value) is not PortfolioCardViewTransform
                for value in self.transforms
            )
        ):
            raise TypeError(
                "transforms must be a non-empty exact tuple of "
                "PortfolioCardViewTransform values"
            )
        canonical = tuple(sorted(set(self.transforms), key=lambda item: item.value))
        if self.transforms != canonical:
            raise ValueError("transforms must be unique and canonically ordered")
        for name in (
            "derived_prompt_view_sha256",
            "derived_evidence_sha256",
            "derived_score_state_sha256",
            "derived_action_evidence_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in (
            "prompt_source_binding_sha256",
            "evidence_source_binding_sha256",
            "score_source_binding_sha256",
            "action_evidence_source_binding_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                require_sha256(value, name)
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        transform_set = set(self.transforms)
        prompt_permutation = (
            PortfolioCardViewTransform.PROMPT_PERMUTATION in transform_set
        )
        evidence_permutation = (
            PortfolioCardViewTransform.EVIDENCE_PERMUTATION in transform_set
        )
        score_permutation = (
            PortfolioCardViewTransform.SCORE_PERMUTATION in transform_set
        )
        action_evidence_permutation = (
            PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION in transform_set
        )
        if prompt_permutation != (self.prompt_source_binding_sha256 is not None):
            raise ValueError(
                "prompt permutation requires exactly one prompt source binding"
            )
        if evidence_permutation != (self.evidence_source_binding_sha256 is not None):
            raise ValueError(
                "evidence permutation requires exactly one evidence source binding"
            )
        if score_permutation != (self.score_source_binding_sha256 is not None):
            raise ValueError(
                "score permutation requires exactly one score source binding"
            )
        if action_evidence_permutation != (
            self.action_evidence_source_binding_sha256 is not None
        ):
            raise ValueError(
                "action-evidence permutation requires exactly one source binding"
            )
        for value in (
            self.prompt_source_binding_sha256,
            self.evidence_source_binding_sha256,
            self.score_source_binding_sha256,
            self.action_evidence_source_binding_sha256,
        ):
            if value is not None and value == self.source_binding_sha256:
                raise ValueError("a permuted view must name a different source binding")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "source_binding_sha256": self.source_binding_sha256,
            "transforms": [value.value for value in self.transforms],
            "derived_prompt_view_sha256": self.derived_prompt_view_sha256,
            "derived_evidence_sha256": self.derived_evidence_sha256,
            "derived_score_state_sha256": self.derived_score_state_sha256,
            "derived_action_evidence_sha256": (self.derived_action_evidence_sha256),
            "prompt_source_binding_sha256": (self.prompt_source_binding_sha256),
            "evidence_source_binding_sha256": (self.evidence_source_binding_sha256),
            "score_source_binding_sha256": self.score_source_binding_sha256,
            "action_evidence_source_binding_sha256": (
                self.action_evidence_source_binding_sha256
            ),
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_CARD_DERIVED_VIEW_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


@dataclass(frozen=True, slots=True)
class CardTransferAdjudicationRequest:
    """Benchmark-owned prediction/outcome projection presented for scoring."""

    card_key: str
    reference: InsightRef
    prediction: FrozenJsonObject
    outcome: FrozenJsonObject
    source_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed lowercase token grammar")
        _reference_record(self.reference)
        for name in ("prediction", "outcome"):
            value = getattr(self, name)
            if type(value) is not FrozenJsonObject:
                raise TypeError(f"{name} must be an exact FrozenJsonObject")
            if freeze_json(value) is not value:
                raise TypeError(f"{name} must already be frozen typed JSON")
        if type(self.source_receipt_sha256s) is not tuple or any(
            type(value) is not str for value in self.source_receipt_sha256s
        ):
            raise TypeError("source_receipt_sha256s must be an exact tuple of strings")
        for value in self.source_receipt_sha256s:
            require_sha256(value, "source_receipt_sha256")
        if self.source_receipt_sha256s != tuple(
            sorted(set(self.source_receipt_sha256s))
        ):
            raise ValueError("source_receipt_sha256s must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "card_key": self.card_key,
            "reference": _reference_record(self.reference),
            "prediction_sha256": typed_json_sha256(self.prediction),
            "outcome_sha256": typed_json_sha256(self.outcome),
            "source_receipt_sha256s": list(self.source_receipt_sha256s),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_TRANSFER_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True, eq=False)
class CardTransferScoreReceipt:
    """Sealed adjudicator output for one exact transfer observation."""

    request_sha256: str
    score_component: CardScoreComponent
    adjudicator_id: str
    adjudicator_version: int
    adjudicator_definition_sha256: str

    def __post_init__(self) -> None:
        require_sha256(self.request_sha256, "request_sha256")
        if type(self.score_component) is not CardScoreComponent:
            raise TypeError("score_component must be an exact CardScoreComponent")
        self.score_component.__post_init__()
        if (
            type(self.adjudicator_id) is not str
            or _TOKEN.fullmatch(self.adjudicator_id) is None
        ):
            raise ValueError(
                "adjudicator_id must use the closed lowercase token grammar"
            )
        if type(self.adjudicator_version) is not int or self.adjudicator_version <= 0:
            raise ValueError("adjudicator_version must be a positive exact integer")
        require_sha256(
            self.adjudicator_definition_sha256,
            "adjudicator_definition_sha256",
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "score_component": self.score_component.to_record(),
            "adjudicator_id": self.adjudicator_id,
            "adjudicator_version": self.adjudicator_version,
            "adjudicator_definition_sha256": self.adjudicator_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_TRANSFER_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "receipt_sha256": self.receipt_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is CardTransferScoreReceipt
            and type(other) is CardTransferScoreReceipt
            and self.receipt_sha256 == other.receipt_sha256
        )

    __hash__ = None


def validate_card_transfer_score_receipt(
    request: CardTransferAdjudicationRequest,
    receipt: CardTransferScoreReceipt,
) -> None:
    """Reject a score receipt detached from its projection or lineage."""

    if type(request) is not CardTransferAdjudicationRequest:
        raise TypeError("request must be an exact CardTransferAdjudicationRequest")
    request.__post_init__()
    if type(receipt) is not CardTransferScoreReceipt:
        raise TypeError("receipt must be an exact CardTransferScoreReceipt")
    receipt.__post_init__()
    if receipt.request_sha256 != request.request_sha256:
        raise ValueError("score receipt is bound to a different transfer request")
    if receipt.score_component.receipt_sha256s != request.source_receipt_sha256s:
        raise ValueError("score component lineage differs from the transfer request")


@runtime_checkable
class CardTransferAdjudicator(Protocol):
    """Benchmark-injected policy for scoring one card transfer observation."""

    def adjudicate(
        self,
        request: CardTransferAdjudicationRequest,
    ) -> CardTransferScoreReceipt: ...


@dataclass(frozen=True, slots=True)
class PortfolioCard:
    """One prompt-visible card view bound to immutable source evidence.

    ``prompt_payload`` is the action-neutral, experiment-controlled view.
    Adaptive, score-permuted, names-only, and redacted arms may project views
    while retaining the same source content/evidence hashes.  The selector
    addresses cards only by the opaque task-local ``card_key``.
    ``finite_action_evidence`` is the prompt-visible action-attribution view.
    On a source card it equals the immutable action lineage in
    ``source_binding``.  A derived experimental view may permute or redact it
    only through an explicit donor-bound view receipt; source provenance stays
    unchanged. Empty evidence preserves legacy card records and prompt bytes.
    """

    card_key: str
    reference: InsightRef
    content_sha256: str
    evidence_sha256: str
    prompt_payload: FrozenJsonObject
    score_components: tuple[CardScoreComponent, ...] = ()
    assigned_score: float | None = None
    finite_action_evidence: tuple[FiniteActionEvidenceBinding, ...] = ()
    source_binding: PortfolioCardSourceBinding | None = None
    derived_view_receipt: PortfolioCardViewReceipt | None = None

    def __post_init__(self) -> None:
        if type(self.card_key) is not str or _TOKEN.fullmatch(self.card_key) is None:
            raise ValueError("card_key must use the closed lowercase token grammar")
        _reference_record(self.reference)
        require_sha256(self.content_sha256, "content_sha256")
        require_sha256(self.evidence_sha256, "evidence_sha256")
        if type(self.prompt_payload) is not FrozenJsonObject:
            raise TypeError("prompt_payload must be an exact FrozenJsonObject")
        if freeze_json(self.prompt_payload) is not self.prompt_payload:
            raise TypeError("prompt_payload must already be frozen typed JSON")
        prompt_view_sha256 = typed_json_sha256(self.prompt_payload)
        score_state_sha256 = portfolio_card_score_state_sha256(
            self.score_components,
            self.assigned_score,
        )
        action_evidence_sha256 = portfolio_card_action_evidence_sha256(
            self.finite_action_evidence
        )
        _validate_finite_action_evidence(
            self.finite_action_evidence,
            name="finite_action_evidence",
        )
        if self.source_binding is None:
            if self.finite_action_evidence:
                raise ValueError(
                    "non-empty finite_action_evidence requires a source binding"
                )
            if self.derived_view_receipt is not None:
                raise ValueError("a derived view receipt requires a source binding")
            return
        if type(self.source_binding) is not PortfolioCardSourceBinding:
            raise TypeError(
                "source_binding must be an exact PortfolioCardSourceBinding or None"
            )
        self.source_binding.__post_init__()
        if (
            self.reference != self.source_binding.reference
            or self.content_sha256 != self.source_binding.content_sha256
        ):
            raise ValueError("card identity differs from its source binding")
        if self.derived_view_receipt is None:
            if (
                prompt_view_sha256 != self.source_binding.source_prompt_view_sha256
                or self.evidence_sha256 != self.source_binding.source_evidence_sha256
                or score_state_sha256 != self.source_binding.source_score_state_sha256
                or self.finite_action_evidence
                != self.source_binding.finite_action_evidence
            ):
                raise ValueError(
                    "card view differs from its source without a derived view receipt"
                )
            return
        if type(self.derived_view_receipt) is not PortfolioCardViewReceipt:
            raise TypeError(
                "derived_view_receipt must be an exact PortfolioCardViewReceipt or None"
            )
        receipt = self.derived_view_receipt
        receipt.__post_init__()
        if receipt.source_binding_sha256 != self.source_binding.binding_sha256:
            raise ValueError("derived view receipt names a different source binding")
        if (
            receipt.derived_prompt_view_sha256 != prompt_view_sha256
            or receipt.derived_evidence_sha256 != self.evidence_sha256
            or receipt.derived_score_state_sha256 != score_state_sha256
            or receipt.derived_action_evidence_sha256 != action_evidence_sha256
        ):
            raise ValueError("derived view receipt differs from the exact card view")
        transforms = set(receipt.transforms)
        prompt_changed = (
            prompt_view_sha256 != self.source_binding.source_prompt_view_sha256
        )
        evidence_changed = (
            self.evidence_sha256 != self.source_binding.source_evidence_sha256
        )
        score_changed = (
            score_state_sha256 != self.source_binding.source_score_state_sha256
        )
        action_evidence_changed = (
            self.finite_action_evidence != self.source_binding.finite_action_evidence
        )
        prompt_transforms = transforms & {
            PortfolioCardViewTransform.PROMPT_PERMUTATION,
            PortfolioCardViewTransform.PROMPT_PROJECTION,
            PortfolioCardViewTransform.PROMPT_REDACTION,
        }
        if len(prompt_transforms) > 1 or (prompt_changed and not prompt_transforms):
            raise ValueError("prompt transform declaration does not match the view")
        if (
            not prompt_changed
            and PortfolioCardViewTransform.PROMPT_PROJECTION in prompt_transforms
        ):
            raise ValueError("prompt projection must change the exact prompt view")
        evidence_transforms = transforms & {
            PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
            PortfolioCardViewTransform.EVIDENCE_REDACTION,
        }
        if len(evidence_transforms) > 1 or (
            evidence_changed and not evidence_transforms
        ):
            raise ValueError("evidence transform declaration does not match the view")
        if (
            PortfolioCardViewTransform.PROMPT_REDACTION in prompt_transforms
            and self.prompt_payload != CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD
        ):
            raise ValueError(
                "redacted prompt payload must use the canonical neutral value"
            )
        if (
            PortfolioCardViewTransform.EVIDENCE_REDACTION in evidence_transforms
            and self.evidence_sha256 != CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256
        ):
            raise ValueError("redacted evidence must use the canonical sentinel")
        score_transforms = transforms & {
            PortfolioCardViewTransform.SCORE_PERMUTATION,
            PortfolioCardViewTransform.SCORE_REDACTION,
        }
        if len(score_transforms) > 1 or (score_changed and not score_transforms):
            raise ValueError("score transform declaration does not match the view")
        if PortfolioCardViewTransform.SCORE_REDACTION in score_transforms and (
            self.score_components or self.assigned_score is not None
        ):
            raise ValueError("redacted score state must be empty")
        action_evidence_transforms = transforms & {
            PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION,
            PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION,
        }
        if len(action_evidence_transforms) > 1 or (
            action_evidence_changed and not action_evidence_transforms
        ):
            raise ValueError(
                "action-evidence transform declaration does not match the view"
            )
        if (
            PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION
            in action_evidence_transforms
            and self.finite_action_evidence
        ):
            raise ValueError("redacted action evidence must be empty")

    @property
    def prompt_view_sha256(self) -> str:
        self.__post_init__()
        return typed_json_sha256(self.prompt_payload)

    @property
    def typed_prompt_payload(self) -> PortfolioCardPromptPayload:
        """Return the scientific split between neutral prose and attribution."""

        self.__post_init__()
        return PortfolioCardPromptPayload(
            action_neutral_payload=self.prompt_payload,
            finite_action_evidence=self.finite_action_evidence,
        )

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "card_key": self.card_key,
            "reference": _reference_record(self.reference),
            "content_sha256": self.content_sha256,
            "evidence_sha256": self.evidence_sha256,
            "prompt_view_sha256": self.prompt_view_sha256,
            "score_components": [
                component.to_record() for component in self.score_components
            ],
            "assigned_score_hex": (
                None if self.assigned_score is None else self.assigned_score.hex()
            ),
            **(
                {}
                if not self.finite_action_evidence
                else {
                    "finite_action_evidence": [
                        binding.to_record() for binding in self.finite_action_evidence
                    ]
                }
            ),
            **(
                {}
                if self.source_binding is None
                else {"source_binding": self.source_binding.to_record()}
            ),
            **(
                {}
                if self.derived_view_receipt is None
                else {"derived_view_receipt": self.derived_view_receipt.to_record()}
            ),
        }

    def prompt_record(self) -> dict[str, object]:
        """Return the exact detached view rendered for the selector."""

        self.__post_init__()
        return {
            "card_key": self.card_key,
            "insight_id": self.reference.insight_id.value,
            "insight_version": self.reference.version,
            "content_sha256": self.content_sha256,
            "evidence_sha256": self.evidence_sha256,
            "score_components": [
                component.prompt_record() for component in self.score_components
            ],
            "assigned_score": self.assigned_score,
            "payload": thaw_json(self.prompt_payload),
            **(
                {}
                if not self.finite_action_evidence
                else {
                    "finite_action_evidence": [
                        binding.to_record() for binding in self.finite_action_evidence
                    ]
                }
            ),
        }


def portfolio_card_snapshot_sha256(cards: tuple[PortfolioCard, ...]) -> str:
    """Bind the complete source/view/score state supplied to one selector."""

    if type(cards) is not tuple or any(
        type(card) is not PortfolioCard for card in cards
    ):
        raise TypeError("cards must be an exact tuple of PortfolioCard values")
    for card in cards:
        card.__post_init__()
    return _hash(
        _CARD_SNAPSHOT_DOMAIN,
        {"cards": [card.to_record() for card in cards]},
    )


def derive_portfolio_card_view(
    source_card: PortfolioCard,
    *,
    prompt_payload: FrozenJsonObject,
    evidence_sha256: str,
    score_components: tuple[CardScoreComponent, ...],
    assigned_score: float | None,
    transforms: tuple[PortfolioCardViewTransform, ...],
    policy_id: str,
    policy_version: int,
    policy_definition_sha256: str,
    prompt_source_card: PortfolioCard | None = None,
    evidence_source_card: PortfolioCard | None = None,
    score_source_card: PortfolioCard | None = None,
    finite_action_evidence: tuple[FiniteActionEvidenceBinding, ...] | None = None,
    action_evidence_source_card: PortfolioCard | None = None,
) -> PortfolioCard:
    """Derive an explicit experimental view without changing source lineage."""

    if type(source_card) is not PortfolioCard:
        raise TypeError("source_card must be an exact PortfolioCard")
    source_card.__post_init__()
    if source_card.source_binding is None:
        raise ValueError("derived views require a source-bound card")
    if source_card.derived_view_receipt is not None:
        raise ValueError(
            "derived views must start from the integrity-bound source view"
        )
    if type(prompt_payload) is not FrozenJsonObject:
        raise TypeError("prompt_payload must be an exact FrozenJsonObject")
    if freeze_json(prompt_payload) is not prompt_payload:
        raise TypeError("prompt_payload must already be frozen typed JSON")
    prompt_source_binding_sha256: str | None = None
    if prompt_source_card is not None:
        if type(prompt_source_card) is not PortfolioCard:
            raise TypeError("prompt_source_card must be an exact PortfolioCard")
        prompt_source_card.__post_init__()
        if prompt_source_card.source_binding is None:
            raise ValueError("prompt source card lacks integrity-bound provenance")
        prompt_source_binding_sha256 = prompt_source_card.source_binding.binding_sha256
        if (
            PortfolioCardViewTransform.PROMPT_PERMUTATION in transforms
            and typed_json_sha256(prompt_payload)
            != prompt_source_card.source_binding.source_prompt_view_sha256
        ):
            raise ValueError("permuted prompt must equal the named source prompt view")
    evidence_source_binding_sha256: str | None = None
    if evidence_source_card is not None:
        if type(evidence_source_card) is not PortfolioCard:
            raise TypeError("evidence_source_card must be an exact PortfolioCard")
        evidence_source_card.__post_init__()
        if evidence_source_card.source_binding is None:
            raise ValueError("evidence source card lacks integrity-bound provenance")
        evidence_source_binding_sha256 = (
            evidence_source_card.source_binding.binding_sha256
        )
        if (
            PortfolioCardViewTransform.EVIDENCE_PERMUTATION in transforms
            and evidence_sha256
            != evidence_source_card.source_binding.source_evidence_sha256
        ):
            raise ValueError("permuted evidence must equal the named source evidence")
    score_source_binding_sha256: str | None = None
    if score_source_card is not None:
        if type(score_source_card) is not PortfolioCard:
            raise TypeError("score_source_card must be an exact PortfolioCard")
        score_source_card.__post_init__()
        if score_source_card.source_binding is None:
            raise ValueError("score source card lacks integrity-bound provenance")
        score_source_binding_sha256 = score_source_card.source_binding.binding_sha256
        if (
            PortfolioCardViewTransform.SCORE_PERMUTATION in transforms
            and portfolio_card_score_state_sha256(
                score_components,
                assigned_score,
            )
            != score_source_card.source_binding.source_score_state_sha256
        ):
            raise ValueError(
                "permuted score state must equal the named source score state"
            )
    projected_action_evidence = (
        source_card.source_binding.finite_action_evidence
        if finite_action_evidence is None
        else finite_action_evidence
    )
    _validate_finite_action_evidence(
        projected_action_evidence,
        name="finite_action_evidence",
    )
    action_evidence_source_binding_sha256: str | None = None
    if action_evidence_source_card is not None:
        if type(action_evidence_source_card) is not PortfolioCard:
            raise TypeError(
                "action_evidence_source_card must be an exact PortfolioCard"
            )
        action_evidence_source_card.__post_init__()
        if action_evidence_source_card.source_binding is None:
            raise ValueError(
                "action-evidence source card lacks integrity-bound provenance"
            )
        action_evidence_source_binding_sha256 = (
            action_evidence_source_card.source_binding.binding_sha256
        )
        if (
            PortfolioCardViewTransform.ACTION_EVIDENCE_PERMUTATION in transforms
            and projected_action_evidence
            != action_evidence_source_card.source_binding.finite_action_evidence
        ):
            raise ValueError(
                "permuted action evidence must equal the named source action evidence"
            )
    if (
        PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION in transforms
        and projected_action_evidence
    ):
        raise ValueError("redacted action evidence must be empty")
    receipt = PortfolioCardViewReceipt(
        source_binding_sha256=source_card.source_binding.binding_sha256,
        transforms=transforms,
        derived_prompt_view_sha256=typed_json_sha256(prompt_payload),
        derived_evidence_sha256=evidence_sha256,
        derived_score_state_sha256=portfolio_card_score_state_sha256(
            score_components,
            assigned_score,
        ),
        derived_action_evidence_sha256=(
            portfolio_card_action_evidence_sha256(projected_action_evidence)
        ),
        prompt_source_binding_sha256=prompt_source_binding_sha256,
        evidence_source_binding_sha256=evidence_source_binding_sha256,
        score_source_binding_sha256=score_source_binding_sha256,
        action_evidence_source_binding_sha256=(action_evidence_source_binding_sha256),
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=policy_definition_sha256,
    )
    return PortfolioCard(
        card_key=source_card.card_key,
        reference=source_card.reference,
        content_sha256=source_card.content_sha256,
        evidence_sha256=evidence_sha256,
        prompt_payload=prompt_payload,
        score_components=score_components,
        assigned_score=assigned_score,
        finite_action_evidence=projected_action_evidence,
        source_binding=source_card.source_binding,
        derived_view_receipt=receipt,
    )


class PortfolioExperimentalArm(str, Enum):
    """Closed scientific M/P/N portfolio-view arms."""

    MEMORY = "m"
    PERMUTED_PLACEBO = "p"
    NEUTRAL = "n"


@dataclass(frozen=True, slots=True)
class PortfolioExperimentalViewReceipt:
    """Request-level commitment to one complete scientific card view.

    Per-card receipts bind compartment bytes.  This receipt additionally binds
    the arm and the population-level permutation invariant, which cannot be
    established by validating cards independently.
    """

    arm: PortfolioExperimentalArm
    source_registry_sha256: str
    card_snapshot_sha256: str
    source_donor_binding_pairs: tuple[tuple[str, str], ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str

    def __post_init__(self) -> None:
        if type(self.arm) is not PortfolioExperimentalArm:
            raise TypeError("arm must be an exact PortfolioExperimentalArm")
        require_sha256(self.source_registry_sha256, "source_registry_sha256")
        require_sha256(self.card_snapshot_sha256, "card_snapshot_sha256")
        if type(self.source_donor_binding_pairs) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or any(type(value) is not str for value in pair)
            for pair in self.source_donor_binding_pairs
        ):
            raise TypeError(
                "source_donor_binding_pairs must be an exact tuple of SHA pairs"
            )
        for source_sha256, donor_sha256 in self.source_donor_binding_pairs:
            require_sha256(source_sha256, "source_binding_sha256")
            require_sha256(donor_sha256, "donor_binding_sha256")
        if self.source_donor_binding_pairs != tuple(
            sorted(self.source_donor_binding_pairs)
        ):
            raise ValueError("source-donor pairs must use canonical source order")
        sources = tuple(pair[0] for pair in self.source_donor_binding_pairs)
        donors = tuple(pair[1] for pair in self.source_donor_binding_pairs)
        if len(set(sources)) != len(sources):
            raise ValueError("source-donor pairs cannot repeat a source")
        if self.arm is PortfolioExperimentalArm.PERMUTED_PLACEBO:
            if not sources:
                raise ValueError("P requires a non-empty donor permutation")
            if set(sources) != set(donors) or len(set(donors)) != len(donors):
                raise ValueError("P donors must be a bijection over source cards")
            if any(
                source == donor for source, donor in self.source_donor_binding_pairs
            ):
                raise ValueError("P donor permutation must be a derangement")
        elif self.source_donor_binding_pairs:
            raise ValueError("only P may carry source-donor pairs")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        require_sha256(self.policy_definition_sha256, "policy_definition_sha256")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "arm": self.arm.value,
            "source_registry_sha256": self.source_registry_sha256,
            "card_snapshot_sha256": self.card_snapshot_sha256,
            "source_donor_binding_pairs": [
                {"source_binding_sha256": source, "donor_binding_sha256": donor}
                for source, donor in self.source_donor_binding_pairs
            ],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash(_EXPERIMENTAL_VIEW_RECEIPT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_sha256": self.receipt_sha256}


def validate_portfolio_experimental_view(
    *,
    cards: tuple[PortfolioCard, ...],
    finite_variation_contract: FiniteVariationContract,
    source_registry: PortfolioCardSourceRegistry,
    receipt: PortfolioExperimentalViewReceipt,
) -> None:
    """Validate a complete provider-neutral M/P/N scientific prompt view."""

    if (
        type(cards) is not tuple
        or not cards
        or any(type(card) is not PortfolioCard for card in cards)
    ):
        raise TypeError("cards must be a non-empty exact tuple of PortfolioCard values")
    identity_index = validated_finite_variation_identity_index(
        finite_variation_contract
    )
    if type(source_registry) is not PortfolioCardSourceRegistry:
        raise TypeError("source_registry must be an exact PortfolioCardSourceRegistry")
    source_registry.__post_init__()
    if type(receipt) is not PortfolioExperimentalViewReceipt:
        raise TypeError("receipt must be an exact PortfolioExperimentalViewReceipt")
    receipt.__post_init__()
    if receipt.source_registry_sha256 != source_registry.registry_sha256:
        raise ValueError("experimental receipt names a different source registry")
    if receipt.card_snapshot_sha256 != portfolio_card_snapshot_sha256(cards):
        raise ValueError("experimental receipt names a different card snapshot")

    sources = {
        binding.binding_sha256: binding for binding in source_registry.source_bindings
    }
    cards_by_source: dict[str, PortfolioCard] = {}
    for card in cards:
        card.__post_init__()
        if card.source_binding is None:
            raise ValueError("scientific views reject unbound legacy cards")
        source_sha256 = card.source_binding.binding_sha256
        if source_sha256 in cards_by_source:
            raise ValueError("scientific views cannot repeat a source card")
        cards_by_source[source_sha256] = card
        admitted = sources.get(source_sha256)
        if admitted is None or admitted != card.source_binding:
            raise ValueError(
                "scientific card source differs from the admitted registry"
            )
        _validate_action_neutral_payload(
            card.typed_prompt_payload,
            identity_index=identity_index,
        )
    if cards_by_source.keys() != sources.keys():
        raise ValueError("scientific cards differ from the complete source registry")

    if receipt.arm is PortfolioExperimentalArm.MEMORY:
        if any(card.derived_view_receipt is not None for card in cards):
            raise ValueError("M requires pristine coherent source cards")
        return

    if receipt.arm is PortfolioExperimentalArm.PERMUTED_PLACEBO:
        required_transforms = tuple(
            sorted(
                (
                    PortfolioCardViewTransform.EVIDENCE_PERMUTATION,
                    PortfolioCardViewTransform.PROMPT_PERMUTATION,
                    PortfolioCardViewTransform.SCORE_PERMUTATION,
                ),
                key=lambda value: value.value,
            )
        )
        inferred_pairs: list[tuple[str, str]] = []
        for source_sha256, card in cards_by_source.items():
            view = card.derived_view_receipt
            if view is None or view.transforms != required_transforms:
                raise ValueError(
                    "P requires exact prompt/evidence/score permutation while "
                    "retaining source action evidence"
                )
            donors = (
                view.prompt_source_binding_sha256,
                view.evidence_source_binding_sha256,
                view.score_source_binding_sha256,
            )
            if any(donor is None for donor in donors) or len(set(donors)) != 1:
                raise ValueError(
                    "P requires one donor across prompt/evidence/score compartments"
                )
            donor_sha256 = donors[0]
            assert donor_sha256 is not None
            donor = sources.get(donor_sha256)
            if donor is None:
                raise ValueError("P donor is outside the admitted source registry")
            if (
                view.derived_prompt_view_sha256 != donor.source_prompt_view_sha256
                or view.derived_evidence_sha256 != donor.source_evidence_sha256
                or view.derived_score_state_sha256 != donor.source_score_state_sha256
            ):
                raise ValueError(
                    "P prompt/evidence/score compartment differs from its donor"
                )
            if (
                view.action_evidence_source_binding_sha256 is not None
                or card.finite_action_evidence
                != card.source_binding.finite_action_evidence
                or view.derived_action_evidence_sha256
                != portfolio_card_action_evidence_sha256(
                    card.source_binding.finite_action_evidence
                )
            ):
                raise ValueError(
                    "P must retain the source card's finite-action evidence"
                )
            inferred_pairs.append((source_sha256, donor_sha256))
        if tuple(sorted(inferred_pairs)) != receipt.source_donor_binding_pairs:
            raise ValueError("P receipt differs from the exact donor permutation")
        return

    required_transforms = tuple(
        sorted(
            (
                PortfolioCardViewTransform.ACTION_EVIDENCE_REDACTION,
                PortfolioCardViewTransform.EVIDENCE_REDACTION,
                PortfolioCardViewTransform.PROMPT_REDACTION,
                PortfolioCardViewTransform.SCORE_REDACTION,
            ),
            key=lambda value: value.value,
        )
    )
    for card in cards:
        view = card.derived_view_receipt
        if view is None or view.transforms != required_transforms:
            raise ValueError("N requires exact canonical compartment redaction")
        if (
            card.prompt_payload != CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD
            or card.evidence_sha256 != CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256
            or card.score_components
            or card.assigned_score is not None
            or card.finite_action_evidence
        ):
            raise ValueError("N card compartments are not canonically neutral")


@dataclass(frozen=True, slots=True)
class _PortfolioSelectionRequestValidationReceipt:
    finite_identity_index: ValidatedFiniteVariationIdentityIndex
    context_sha256: str
    card_snapshot_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.finite_identity_index)
            is not ValidatedFiniteVariationIdentityIndex
        ):
            raise TypeError("finite_identity_index must be exact")
        self.finite_identity_index.__post_init__()
        require_sha256(self.context_sha256, "context_sha256")
        require_sha256(self.card_snapshot_sha256, "card_snapshot_sha256")


@dataclass(frozen=True, slots=True)
class PortfolioSelectionRequest:
    """One exact logical call over a sealed action and card snapshot."""

    call_id: LLMCallId
    operation: str
    instruction: str
    context: FrozenJsonObject
    finite_variation_contract: FiniteVariationContract
    cards: tuple[PortfolioCard, ...]
    portfolio_size: int
    required_metric_ids: tuple[str, ...]
    min_distinct_families: int | None = None
    require_supporting_cards: bool = True
    require_pairwise_disjoint_parent_patches: bool = False
    max_output_tokens: int = 2_048
    temperature: float | None = None
    source_registry: PortfolioCardSourceRegistry | None = None
    experimental_view_receipt: PortfolioExperimentalViewReceipt | None = None
    memory_dose_contract: BoundedPortfolioMemoryDoseContract | None = None
    candidate_pool_required_option_ids: tuple[str, ...] = ()
    _validation_receipt: _PortfolioSelectionRequestValidationReceipt = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if type(self.call_id) is not LLMCallId:
            raise TypeError("call_id must be an exact LLMCallId")
        LLMCallId.__post_init__(self.call_id)
        if type(self.operation) is not str or _TOKEN.fullmatch(self.operation) is None:
            raise ValueError("operation must use the closed lowercase token grammar")
        if (
            type(self.instruction) is not str
            or not self.instruction.strip()
            or self.instruction != self.instruction.strip()
        ):
            raise ValueError("instruction must be canonical non-empty text")
        if type(self.context) is not FrozenJsonObject:
            raise TypeError("context must be an exact FrozenJsonObject")
        if freeze_json(self.context) is not self.context:
            raise TypeError("context must already be frozen typed JSON")
        finite_identity_index = validated_finite_variation_identity_index(
            self.finite_variation_contract
        )
        if type(self.cards) is not tuple or not self.cards:
            raise ValueError("cards must be a non-empty exact tuple")
        if any(type(card) is not PortfolioCard for card in self.cards):
            raise TypeError("cards must contain exact PortfolioCard values")
        for card in self.cards:
            card.__post_init__()
        card_keys = tuple(card.card_key for card in self.cards)
        if card_keys != tuple(sorted(set(card_keys))):
            raise ValueError("cards must use unique canonical card_key order")
        references = tuple(card.reference for card in self.cards)
        if len(set(references)) != len(references):
            raise ValueError("cards cannot repeat an exact insight reference")
        source_bindings = {
            card.source_binding.binding_sha256: card.source_binding
            for card in self.cards
            if card.source_binding is not None
        }
        if source_bindings:
            if len(source_bindings) != len(self.cards):
                raise ValueError(
                    "a source-admitted request cannot mix bound and legacy cards"
                )
            if type(self.source_registry) is not PortfolioCardSourceRegistry:
                raise ValueError(
                    "source-bound cards require trusted application admission"
                )
            self.source_registry.__post_init__()
            admitted = {
                binding.binding_sha256: binding
                for binding in self.source_registry.source_bindings
            }
            if admitted.keys() != source_bindings.keys():
                raise ValueError(
                    "source registry differs from the request card source set"
                )
            for binding_sha256, binding in source_bindings.items():
                if admitted[binding_sha256] != binding:
                    raise ValueError(
                        "source registry binding differs from the request card"
                    )
        elif self.source_registry is not None:
            raise ValueError("legacy unbound cards cannot carry a source registry")
        for card in self.cards:
            receipt = card.derived_view_receipt
            if receipt is None:
                continue
            prompt_source_sha256 = receipt.prompt_source_binding_sha256
            if prompt_source_sha256 is not None:
                prompt_source = source_bindings.get(prompt_source_sha256)
                if prompt_source is None:
                    raise ValueError(
                        "derived card view names a prompt source outside the request"
                    )
                if (
                    receipt.derived_prompt_view_sha256
                    != prompt_source.source_prompt_view_sha256
                ):
                    raise ValueError(
                        "permuted prompt differs from the named source view"
                    )
            evidence_source_sha256 = receipt.evidence_source_binding_sha256
            if evidence_source_sha256 is not None:
                evidence_source = source_bindings.get(evidence_source_sha256)
                if evidence_source is None:
                    raise ValueError(
                        "derived card view names an evidence source outside the request"
                    )
                if (
                    receipt.derived_evidence_sha256
                    != evidence_source.source_evidence_sha256
                ):
                    raise ValueError(
                        "permuted evidence differs from the named source view"
                    )
            score_source_sha256 = receipt.score_source_binding_sha256
            if score_source_sha256 is not None:
                score_source = source_bindings.get(score_source_sha256)
                if score_source is None:
                    raise ValueError(
                        "derived card view names a score source outside the request"
                    )
                if (
                    receipt.derived_score_state_sha256
                    != score_source.source_score_state_sha256
                ):
                    raise ValueError(
                        "permuted score state differs from the named source view"
                    )
            action_source_sha256 = receipt.action_evidence_source_binding_sha256
            if action_source_sha256 is not None:
                action_source = source_bindings.get(action_source_sha256)
                if action_source is None:
                    raise ValueError(
                        "derived card view names an action-evidence source "
                        "outside the request"
                    )
                if (
                    receipt.derived_action_evidence_sha256
                    != portfolio_card_action_evidence_sha256(
                        action_source.finite_action_evidence
                    )
                ):
                    raise ValueError(
                        "permuted action evidence differs from the named source view"
                    )
        if self.experimental_view_receipt is not None:
            if (
                type(self.experimental_view_receipt)
                is not PortfolioExperimentalViewReceipt
            ):
                raise TypeError(
                    "experimental_view_receipt must be an exact "
                    "PortfolioExperimentalViewReceipt or None"
                )
            if type(self.source_registry) is not PortfolioCardSourceRegistry:
                raise ValueError(
                    "a scientific experimental view requires source admission"
                )
            validate_portfolio_experimental_view(
                cards=self.cards,
                finite_variation_contract=self.finite_variation_contract,
                source_registry=self.source_registry,
                receipt=self.experimental_view_receipt,
            )
        if type(self.portfolio_size) is not int or self.portfolio_size <= 0:
            raise ValueError("portfolio_size must be a positive exact integer")
        if self.portfolio_size > len(self.finite_variation_contract.options):
            raise ValueError("portfolio_size exceeds the finite option count")
        _canonical_metric_ids(self.required_metric_ids)
        available_families = {
            option.family for option in self.finite_variation_contract.options
        }
        if self.min_distinct_families is not None:
            if (
                type(self.min_distinct_families) is not int
                or self.min_distinct_families <= 0
            ):
                raise ValueError(
                    "min_distinct_families must be a positive exact integer or None"
                )
            if self.min_distinct_families > self.portfolio_size:
                raise ValueError("min_distinct_families cannot exceed portfolio_size")
            if self.min_distinct_families > len(available_families):
                raise ValueError("finite contract cannot satisfy min_distinct_families")
        if type(self.require_supporting_cards) is not bool:
            raise TypeError("require_supporting_cards must be an exact bool")
        if self.memory_dose_contract is not None:
            if type(self.memory_dose_contract) is not (
                BoundedPortfolioMemoryDoseContract
            ):
                raise TypeError(
                    "memory_dose_contract must be an exact bounded contract or None"
                )
            self.memory_dose_contract.__post_init__()
            if self.require_supporting_cards:
                raise ValueError(
                    "bounded memory dose requires unattributed exploration slots"
                )
            if (
                self.memory_dose_contract.finite_contract_identity_sha256
                != finite_identity_index.contract_identity_sha256
            ):
                raise ValueError("memory-dose contract names a foreign finite contract")
            cards_by_key = {value.card_key: value for value in self.cards}
            if not set(self.memory_dose_contract.assigned_card_keys).issubset(
                cards_by_key
            ):
                raise ValueError(
                    "memory-dose contract names a card outside the request"
                )
            if any(
                cards_by_key[support.card_key].content_sha256
                != support.card_content_sha256
                for support in self.memory_dose_contract.card_supports
            ):
                raise ValueError(
                    "memory-dose support differs from the request card content"
                )
        if (
            type(self.candidate_pool_required_option_ids) is not tuple
            or any(
                type(value) is not str or _OPTION_ID.fullmatch(value) is None
                for value in self.candidate_pool_required_option_ids
            )
        ):
            raise TypeError(
                "candidate_pool_required_option_ids must be an exact option-ID tuple"
            )
        if self.candidate_pool_required_option_ids != tuple(
            sorted(set(self.candidate_pool_required_option_ids))
        ):
            raise ValueError(
                "candidate_pool_required_option_ids must be unique and canonical"
            )
        if not set(self.candidate_pool_required_option_ids).issubset(
            finite_identity_index.option_ids
        ):
            raise ValueError(
                "candidate_pool_required_option_ids escapes the finite contract"
            )
        if type(self.require_pairwise_disjoint_parent_patches) is not bool:
            raise TypeError(
                "require_pairwise_disjoint_parent_patches must be an exact bool"
            )
        if self.require_pairwise_disjoint_parent_patches and not (
            finite_portfolio_has_pairwise_disjoint_parent_patches(
                self.finite_variation_contract,
                portfolio_size=self.portfolio_size,
                min_distinct_families=self.min_distinct_families,
            )
        ):
            raise ValueError(
                "finite contract has no feasible pairwise-disjoint portfolio"
            )
        if (
            type(self.max_output_tokens) is not int
            or not 1 <= self.max_output_tokens <= MAX_OUTPUT_TOKENS
        ):
            raise ValueError(f"max_output_tokens must lie in [1, {MAX_OUTPUT_TOKENS}]")
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0 <= float(self.temperature) <= 2
        ):
            raise ValueError("temperature must be finite in [0,2] or None")
        object.__setattr__(
            self,
            "_validation_receipt",
            _PortfolioSelectionRequestValidationReceipt(
                finite_identity_index=finite_identity_index,
                context_sha256=typed_json_sha256(self.context),
                card_snapshot_sha256=portfolio_card_snapshot_sha256(self.cards),
            ),
        )

    @property
    def context_sha256(self) -> str:
        if type(self.context) is not FrozenJsonObject:
            raise TypeError("context must be an exact FrozenJsonObject")
        if freeze_json(self.context) is not self.context:
            raise TypeError("context must already be frozen typed JSON")
        return typed_json_sha256(self.context)

    @property
    def card_snapshot_sha256(self) -> str:
        return portfolio_card_snapshot_sha256(self.cards)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        receipt = self._validation_receipt
        return {
            "schema_version": 1,
            "call_id": self.call_id.value,
            "operation": self.operation,
            "instruction_sha256": hashlib.sha256(
                self.instruction.encode("utf-8", errors="strict")
            ).hexdigest(),
            "context_sha256": receipt.context_sha256,
            "finite_contract_identity_sha256": (
                receipt.finite_identity_index.contract_identity_sha256
            ),
            "card_snapshot_sha256": receipt.card_snapshot_sha256,
            "portfolio_size": self.portfolio_size,
            "required_metric_ids": list(self.required_metric_ids),
            **(
                {}
                if self.source_registry is None
                else {"source_registry_sha256": self.source_registry.registry_sha256}
            ),
            **(
                {}
                if self.experimental_view_receipt is None
                else {
                    "experimental_view_receipt_sha256": (
                        self.experimental_view_receipt.receipt_sha256
                    )
                }
            ),
            "min_distinct_families": self.min_distinct_families,
            "require_supporting_cards": self.require_supporting_cards,
            **(
                {}
                if self.memory_dose_contract is None
                else {"memory_dose_contract": (self.memory_dose_contract.to_record())}
            ),
            **(
                {}
                if not self.candidate_pool_required_option_ids
                else {
                    "candidate_pool_required_option_ids": list(
                        self.candidate_pool_required_option_ids
                    )
                }
            ),
            **(
                {
                    "require_pairwise_disjoint_parent_patches": True,
                }
                if self.require_pairwise_disjoint_parent_patches
                else {}
            ),
            "max_output_tokens": self.max_output_tokens,
            "temperature_hex": (
                None if self.temperature is None else float(self.temperature).hex()
            ),
        }

    @property
    def request_sha256(self) -> str:
        return _hash(_REQUEST_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class PortfolioMemberDraft:
    """Framework-neutral structured output before trusted option resolution."""

    option_id: str
    supporting_card_keys: tuple[str, ...]
    effect_predictions: tuple[MetricEffectPrediction, ...]
    design_rationale: str

    def __post_init__(self) -> None:
        if (
            type(self.option_id) is not str
            or _OPTION_ID.fullmatch(self.option_id) is None
        ):
            raise ValueError("option_id must use the closed option grammar")
        if type(self.supporting_card_keys) is not tuple or any(
            type(value) is not str or _TOKEN.fullmatch(value) is None
            for value in self.supporting_card_keys
        ):
            raise TypeError(
                "supporting_card_keys must be an exact tuple of card tokens"
            )
        if self.supporting_card_keys != tuple(sorted(set(self.supporting_card_keys))):
            raise ValueError("supporting_card_keys must be unique and canonical")
        if type(self.effect_predictions) is not tuple or any(
            type(value) is not MetricEffectPrediction
            for value in self.effect_predictions
        ):
            raise TypeError(
                "effect_predictions must contain exact MetricEffectPrediction values"
            )
        for prediction in self.effect_predictions:
            MetricEffectPrediction.__post_init__(prediction)
        metric_ids = tuple(value.metric_id for value in self.effect_predictions)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("effect_predictions must be unique and metric-ordered")
        if (
            type(self.design_rationale) is not str
            or not self.design_rationale.strip()
            or self.design_rationale != self.design_rationale.strip()
        ):
            raise ValueError("design_rationale must be canonical non-empty text")


@dataclass(frozen=True, slots=True)
class RankedPortfolioMember:
    """One trusted ranked action resolved against the sealed finite contract."""

    rank: int
    option_id: str
    option_identity_sha256: str
    child_configuration_sha256: str
    family: str
    supporting_card_keys: tuple[str, ...]
    effect_predictions: tuple[MetricEffectPrediction, ...]
    design_rationale: str

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("rank must be a positive exact integer")
        PortfolioMemberDraft(
            option_id=self.option_id,
            supporting_card_keys=self.supporting_card_keys,
            effect_predictions=self.effect_predictions,
            design_rationale=self.design_rationale,
        )
        require_sha256(self.option_identity_sha256, "option_identity_sha256")
        require_sha256(
            self.child_configuration_sha256,
            "child_configuration_sha256",
        )
        if type(self.family) is not str or _TOKEN.fullmatch(self.family) is None:
            raise ValueError("family must use the closed lowercase token grammar")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "rank": self.rank,
            "option_id": self.option_id,
            "option_identity_sha256": self.option_identity_sha256,
            "child_configuration_sha256": self.child_configuration_sha256,
            "family": self.family,
            "supporting_card_keys": list(self.supporting_card_keys),
            "effect_predictions": [
                {
                    "metric_id": prediction.metric_id,
                    "direction": prediction.direction.value,
                }
                for prediction in self.effect_predictions
            ],
            "design_rationale_sha256": hashlib.sha256(
                self.design_rationale.encode("utf-8", errors="strict")
            ).hexdigest(),
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return the committed member record with its rationale plaintext.

        ``to_record`` remains the stable hash boundary: the rationale is bound
        there by digest only.  This projection makes model-authored reasoning
        available to trace analysis while retaining that committed digest for
        direct verification.
        """

        return {**self.to_record(), "design_rationale": self.design_rationale}


@dataclass(frozen=True, slots=True, eq=False)
class RankedPortfolioDecision:
    """All-or-nothing ranked selection bound to one exact request snapshot."""

    request_sha256: str
    context_sha256: str
    finite_contract_identity_sha256: str
    card_snapshot_sha256: str
    members: tuple[RankedPortfolioMember, ...]
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "context_sha256",
            "finite_contract_identity_sha256",
            "card_snapshot_sha256",
            "policy_definition_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.members) is not tuple or not self.members:
            raise ValueError("members must be a non-empty exact tuple")
        if any(type(member) is not RankedPortfolioMember for member in self.members):
            raise TypeError("members must contain exact RankedPortfolioMember values")
        for member in self.members:
            member.__post_init__()
        if tuple(member.rank for member in self.members) != tuple(
            range(1, len(self.members) + 1)
        ):
            raise ValueError("member ranks must be contiguous and tuple-ordered")
        option_ids = tuple(member.option_id for member in self.members)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("a ranked portfolio cannot repeat an option")
        if type(self.policy_id) is not str or _TOKEN.fullmatch(self.policy_id) is None:
            raise ValueError("policy_id must use the closed lowercase token grammar")
        if type(self.policy_version) is not int or self.policy_version <= 0:
            raise ValueError("policy_version must be a positive exact integer")
        if self.memory_dose_assessment is not None:
            if type(self.memory_dose_assessment) is not (PortfolioMemoryDoseAssessment):
                raise TypeError("memory_dose_assessment must be exact or None")
            self.memory_dose_assessment.__post_init__()
            if (
                self.memory_dose_assessment.stage
                is not PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO
                or not self.memory_dose_assessment.passed
            ):
                raise ValueError(
                    "ranked decision memory dose must be a passing evaluation"
                )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "context_sha256": self.context_sha256,
            "finite_contract_identity_sha256": (self.finite_contract_identity_sha256),
            "card_snapshot_sha256": self.card_snapshot_sha256,
            "members": [member.to_record() for member in self.members],
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_definition_sha256": self.policy_definition_sha256,
            **(
                {}
                if self.memory_dose_assessment is None
                else {
                    "memory_dose_assessment": (self.memory_dose_assessment.to_record())
                }
            ),
        }

    @property
    def decision_sha256(self) -> str:
        return _hash(_DECISION_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        record = self._unsigned_record()
        return {**record, "decision_sha256": self.decision_sha256}

    def to_audit_record(self) -> dict[str, object]:
        """Return the decision commitment plus auditable member plaintext."""

        record = self.to_record()
        return {
            **record,
            "members": [member.to_audit_record() for member in self.members],
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(self) is RankedPortfolioDecision
            and type(other) is RankedPortfolioDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


def resolve_ranked_portfolio_decision(
    request: PortfolioSelectionRequest,
    drafts: tuple[PortfolioMemberDraft, ...],
    *,
    policy_id: str,
    policy_version: int,
    policy_definition_sha256: str,
    memory_dose_assessment: PortfolioMemoryDoseAssessment | None = None,
) -> RankedPortfolioDecision:
    """Resolve the complete structured output or publish no decision."""

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be an exact PortfolioSelectionRequest")
    request.__post_init__()
    if type(drafts) is not tuple or any(
        type(draft) is not PortfolioMemberDraft for draft in drafts
    ):
        raise TypeError("drafts must be an exact tuple of PortfolioMemberDraft values")
    if len(drafts) != request.portfolio_size:
        raise ValueError(
            "selector output does not contain exactly portfolio_size members"
        )
    for draft in drafts:
        draft.__post_init__()
    option_ids = tuple(draft.option_id for draft in drafts)
    if len(set(option_ids)) != len(option_ids):
        raise ValueError("selector output repeats a finite option")
    card_keys = {card.card_key for card in request.cards}
    for draft in drafts:
        if not set(draft.supporting_card_keys).issubset(card_keys):
            raise ValueError(
                "selector output cites a card outside the request snapshot"
            )
        if request.require_supporting_cards and not draft.supporting_card_keys:
            raise ValueError("every portfolio member must cite a supporting card")
        if tuple(value.metric_id for value in draft.effect_predictions) != (
            request.required_metric_ids
        ):
            raise ValueError(
                "every portfolio member must predict the exact required metrics"
            )

    contract = request.finite_variation_contract
    options = tuple(contract.resolve(draft.option_id) for draft in drafts)
    if (
        request.min_distinct_families is not None
        and len({option.family for option in options}) < request.min_distinct_families
    ):
        raise ValueError("selector output violates min_distinct_families")
    if request.require_pairwise_disjoint_parent_patches:
        validate_pairwise_disjoint_parent_patch_selection(
            contract,
            option_ids,
        )
    decision = RankedPortfolioDecision(
        request_sha256=request.request_sha256,
        context_sha256=request.context_sha256,
        finite_contract_identity_sha256=contract.identity_sha256,
        card_snapshot_sha256=request.card_snapshot_sha256,
        members=tuple(
            RankedPortfolioMember(
                rank=index,
                option_id=option.option_id,
                option_identity_sha256=option.identity_sha256,
                child_configuration_sha256=option.child_configuration_sha256,
                family=option.family,
                supporting_card_keys=draft.supporting_card_keys,
                effect_predictions=draft.effect_predictions,
                design_rationale=draft.design_rationale,
            )
            for index, (draft, option) in enumerate(
                zip(drafts, options, strict=True),
                start=1,
            )
        ),
        policy_id=policy_id,
        policy_version=policy_version,
        policy_definition_sha256=policy_definition_sha256,
        memory_dose_assessment=memory_dose_assessment,
    )
    validate_ranked_portfolio_decision(request, decision)
    return decision


def validate_ranked_portfolio_decision(
    request: PortfolioSelectionRequest,
    decision: RankedPortfolioDecision,
) -> None:
    """Revalidate a decision against the exact request at any trust boundary."""

    if type(request) is not PortfolioSelectionRequest:
        raise TypeError("request must be an exact PortfolioSelectionRequest")
    request.__post_init__()
    if type(decision) is not RankedPortfolioDecision:
        raise TypeError("decision must be an exact RankedPortfolioDecision")
    decision.__post_init__()
    if (
        decision.request_sha256 != request.request_sha256
        or decision.context_sha256 != request.context_sha256
        or decision.finite_contract_identity_sha256
        != request.finite_variation_contract.identity_sha256
        or decision.card_snapshot_sha256 != request.card_snapshot_sha256
    ):
        raise ValueError("decision is bound to a different request snapshot")
    if len(decision.members) != request.portfolio_size:
        raise ValueError("decision member count differs from portfolio_size")
    cards = {card.card_key for card in request.cards}
    option_families: set[str] = set()
    for member in decision.members:
        option = request.finite_variation_contract.resolve(member.option_id)
        if (
            member.option_identity_sha256 != option.identity_sha256
            or member.child_configuration_sha256 != option.child_configuration_sha256
            or member.family != option.family
        ):
            raise ValueError("decision member differs from its sealed finite option")
        if not set(member.supporting_card_keys).issubset(cards):
            raise ValueError("decision member cites a foreign card")
        if request.require_supporting_cards and not member.supporting_card_keys:
            raise ValueError("decision member omitted required card attribution")
        if (
            tuple(prediction.metric_id for prediction in member.effect_predictions)
            != request.required_metric_ids
        ):
            raise ValueError("decision member metric predictions differ from request")
        option_families.add(option.family)
    dose = request.memory_dose_contract
    assessment = decision.memory_dose_assessment
    if dose is None:
        if assessment is not None:
            raise ValueError("decision supplied memory dose for an unbounded request")
    else:
        if assessment is None:
            raise ValueError(
                "bounded request requires an evaluated memory-dose receipt"
            )
        assessment.__post_init__()
        if (
            assessment.contract_sha256 != dose.contract_sha256
            or assessment.stage is not PortfolioMemoryDoseStage.EVALUATED_PORTFOLIO
            or not assessment.passed
        ):
            raise ValueError(
                "decision memory-dose receipt differs from the request contract"
            )
        expected_members = tuple(
            PortfolioMemoryDoseMember(
                rank=value.rank,
                option_id=value.option_id,
                option_identity_sha256=value.option_identity_sha256,
                supporting_card_keys=value.supporting_card_keys,
            )
            for value in decision.members
        )
        if assessment.member_content_binding_sha256s != tuple(
            value.content_binding_sha256 for value in expected_members
        ):
            raise ValueError(
                "decision memory-dose receipt differs from selected members"
            )
    if request.min_distinct_families is not None and len(option_families) < (
        request.min_distinct_families
    ):
        raise ValueError("decision violates min_distinct_families")
    if request.require_pairwise_disjoint_parent_patches:
        validate_pairwise_disjoint_parent_patch_selection(
            request.finite_variation_contract,
            tuple(member.option_id for member in decision.members),
        )


@dataclass(frozen=True, slots=True, eq=False)
class PortfolioSelectionSupplementalAudit:
    """Opaque, integrity-bound selector evidence carried beside a v1 decision.

    The ranked decision and materialization receipt remain the stable evaluator
    boundary.  Optional selectors may retain richer policy-specific evidence in
    ``payload`` without making the application layer depend on that policy.
    """

    audit_kind: str
    request_sha256: str
    decision_sha256: str
    payload: FrozenJsonObject

    def __post_init__(self) -> None:
        if (
            type(self.audit_kind) is not str
            or _TOKEN.fullmatch(self.audit_kind) is None
        ):
            raise ValueError("audit_kind must use the closed lowercase token grammar")
        require_sha256(self.request_sha256, "request_sha256")
        require_sha256(self.decision_sha256, "decision_sha256")
        if type(self.payload) is not FrozenJsonObject:
            raise TypeError("payload must be an exact FrozenJsonObject")
        if freeze_json(self.payload) is not self.payload:
            raise TypeError("payload must already be frozen typed JSON")

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "audit_kind": self.audit_kind,
            "request_sha256": self.request_sha256,
            "decision_sha256": self.decision_sha256,
            "payload_sha256": typed_json_sha256(self.payload),
        }

    @property
    def audit_sha256(self) -> str:
        return _hash(_SUPPLEMENTAL_AUDIT_DOMAIN, self._unsigned_record())

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "payload": thaw_json(self.payload),
            "audit_sha256": self.audit_sha256,
        }

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is PortfolioSelectionSupplementalAudit
            and self.audit_sha256 == other.audit_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class PortfolioSelectionResult:
    """One selection outcome and the provider evidence that produced it.

    ``telemetry`` and ``provider_free`` are mutually exclusive and exactly one
    must be present. A bare ``None`` telemetry with no assertion is rejected:
    absence must be asserted and then measured, never inferred from a missing
    field. A policy that reaches the provider supplies telemetry as before; a
    policy that does not must say so, and the runtime confirms it against the
    outbound journals rather than taking the claim on trust.
    """

    decision: RankedPortfolioDecision
    telemetry: AgenticCallTelemetry | None
    supplemental_audit: PortfolioSelectionSupplementalAudit | None = None
    provider_free: bool = False

    def __post_init__(self) -> None:
        if type(self.decision) is not RankedPortfolioDecision:
            raise TypeError("decision must be an exact RankedPortfolioDecision")
        self.decision.__post_init__()
        if type(self.provider_free) is not bool:
            raise TypeError("provider_free must be an exact bool")
        if self.provider_free and self.telemetry is not None:
            raise ValueError(
                "a provider-free selection cannot also carry call telemetry"
            )
        if not self.provider_free and self.telemetry is None:
            raise ValueError(
                "selection telemetry is absent and provider_free was not "
                "asserted; absence must be asserted, not inferred"
            )
        if self.telemetry is not None:
            if type(self.telemetry) is not AgenticCallTelemetry:
                raise TypeError("telemetry must be exact or None")
            AgenticCallTelemetry.__post_init__(self.telemetry)
        if self.supplemental_audit is not None:
            if type(self.supplemental_audit) is not PortfolioSelectionSupplementalAudit:
                raise TypeError("supplemental_audit must be exact or None")
            self.supplemental_audit.__post_init__()
            if (
                self.supplemental_audit.request_sha256 != self.decision.request_sha256
                or self.supplemental_audit.decision_sha256
                != self.decision.decision_sha256
            ):
                raise ValueError(
                    "supplemental audit differs from the ranked decision identity"
                )


@runtime_checkable
class PortfolioSelectionPolicy(Protocol):
    """Select one complete ranked portfolio; partial results are not representable."""

    async def select(
        self,
        request: PortfolioSelectionRequest,
    ) -> PortfolioSelectionResult: ...


__all__ = [
    "CANONICAL_NEUTRAL_PORTFOLIO_PROMPT_PAYLOAD",
    "CANONICAL_REDACTED_PORTFOLIO_EVIDENCE_SHA256",
    "CardScoreComponent",
    "CardTransferAdjudicationRequest",
    "CardTransferAdjudicator",
    "CardTransferScoreReceipt",
    "PortfolioCard",
    "PortfolioCardPromptPayload",
    "PortfolioCardSourceBinding",
    "PortfolioCardSourceRegistry",
    "PortfolioCardViewReceipt",
    "PortfolioCardViewTransform",
    "PortfolioExperimentalArm",
    "PortfolioExperimentalViewReceipt",
    "PortfolioMemberDraft",
    "PortfolioSelectionPolicy",
    "PortfolioSelectionRequest",
    "PortfolioSelectionResult",
    "PortfolioSelectionSupplementalAudit",
    "RankedPortfolioDecision",
    "RankedPortfolioMember",
    "derive_portfolio_card_view",
    "portfolio_card_action_evidence_sha256",
    "portfolio_card_score_state_sha256",
    "portfolio_card_snapshot_sha256",
    "finite_portfolio_has_pairwise_disjoint_parent_patches",
    "finite_option_ids_have_pairwise_disjoint_parent_patch_subset",
    "pairwise_disjoint_parent_patch_witness",
    "pairwise_disjoint_parent_patch_pairs",
    "project_family_exposure_bounds_to_pairwise_disjoint_feasibility",
    "resolve_ranked_portfolio_decision",
    "validate_card_transfer_score_receipt",
    "validate_portfolio_experimental_view",
    "validate_pairwise_disjoint_parent_patch_selection",
    "validate_ranked_portfolio_decision",
]

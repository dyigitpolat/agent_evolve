"""Workload-neutral structural evidence for a sealed finite action palette.

The calibrated allocator needs option identities, phenotype identities, a
stable locus key, and bounded novelty/coverage scores.  None of those facts
should be invented by the model.  This policy derives them before the provider
call from the exact parent/child trees and an authenticated phenotype ledger.

The score is intentionally modest: phenotype novelty is binary under the
frozen evaluated-phenotype cutoff, while structural coverage is the mean of
inverse-frequency family and changed-path-set rarity inside the current
palette.  It contains no objective values and no workload vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass

from agent_evolve.domain.finite_variation import (
    FiniteVariationContract,
    validate_finite_variation_contract,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.patch import require_sha256
from agent_evolve.policies.selection.calibrated_portfolio_binding import (
    CalibratedPortfolioOptionEvidence,
)
from agent_evolve.policies.selection.calibrated_slate import (
    SlateStructuralEvidence,
)
from agent_evolve.policies.variation.exact_parent_crossover import (
    canonical_candidate_path_text,
)
from agent_evolve.policies.variation.typed_patch import derive_patch


POLICY_ID = "finite_palette_structural_evidence"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:finite-palette-structural-evidence:v1\x00"
    b"locus=sha256(canonical-changed-path-set);"
    b"archive-novelty=phenotype-not-in-frozen-known-set;"
    b"coverage=mean(min-family-count/family-count,"
    b"min-locus-count/locus-count);objective-values=false"
).hexdigest()

_LOCUS_DOMAIN = b"agent-evolve:finite-palette-locus:v1\x00"
_EVIDENCE_DOMAIN = b"agent-evolve:finite-palette-evidence:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _canonical_sha256_pairs(
    values: tuple[tuple[str, str], ...],
    *,
    name: str,
) -> None:
    if type(values) is not tuple or any(
        type(value) is not tuple
        or len(value) != 2
        or type(value[0]) is not str
        or not value[0]
        or type(value[1]) is not str
        for value in values
    ):
        raise TypeError(f"{name} must contain exact option/SHA-256 pairs")
    for _, digest in values:
        require_sha256(digest, f"{name}.sha256")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and canonical")
    if len({option_id for option_id, _ in values}) != len(values):
        raise ValueError(f"{name} cannot repeat an option ID")


@dataclass(frozen=True, slots=True)
class FinitePaletteStructuralEvidenceRequest:
    """Complete pre-provider facts needed to project one finite palette."""

    contract: FiniteVariationContract
    option_phenotype_sha256s: tuple[tuple[str, str], ...]
    known_phenotype_sha256s: tuple[str, ...]
    eligibility_receipt_sha256: str
    frozen_archive_snapshot_sha256: str

    def __post_init__(self) -> None:
        if type(self.contract) is not FiniteVariationContract:
            raise TypeError("contract must be an exact FiniteVariationContract")
        validate_finite_variation_contract(self.contract)
        _canonical_sha256_pairs(
            self.option_phenotype_sha256s,
            name="option_phenotype_sha256s",
        )
        expected = tuple(sorted(option.option_id for option in self.contract.options))
        observed = tuple(option_id for option_id, _ in self.option_phenotype_sha256s)
        if observed != expected:
            raise ValueError("phenotype bindings must cover the exact finite contract")
        if type(self.known_phenotype_sha256s) is not tuple:
            raise TypeError("known_phenotype_sha256s must be an exact tuple")
        for digest in self.known_phenotype_sha256s:
            require_sha256(digest, "known_phenotype_sha256")
        if self.known_phenotype_sha256s != tuple(
            sorted(set(self.known_phenotype_sha256s))
        ):
            raise ValueError("known phenotype identities must be unique and canonical")
        require_sha256(
            self.eligibility_receipt_sha256,
            "eligibility_receipt_sha256",
        )
        require_sha256(
            self.frozen_archive_snapshot_sha256,
            "frozen_archive_snapshot_sha256",
        )


def _changed_path_key(contract: FiniteVariationContract, option_index: int) -> str:
    option = contract.options[option_index]
    patch = derive_patch(
        contract.parent_configuration,
        option.child_configuration,
        base_candidate_id=CandidateId("candidate_palette_parent"),
        target_candidate_id=CandidateId("candidate_palette_child"),
    )
    paths = tuple(
        sorted(
            {
                canonical_candidate_path_text(operation.path)
                for operation in patch.operations
            }
        )
    )
    if not paths:
        raise ValueError("finite option produced no changed candidate path")
    digest = hashlib.sha256(
        _LOCUS_DOMAIN + _canonical_json({"changed_paths": list(paths)})
    ).hexdigest()
    return f"locus.{digest[:32]}"


@dataclass(frozen=True, slots=True)
class FinitePaletteStructuralEvidencePolicy:
    """Project exact option evidence without consulting outcomes or a model."""

    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if self.policy_id != POLICY_ID:
            raise ValueError("policy_id drifted")
        if self.policy_version != POLICY_VERSION:
            raise ValueError("policy_version drifted")
        if self.definition_sha256 != POLICY_DEFINITION_SHA256:
            raise ValueError("definition_sha256 drifted")

    def project(
        self,
        request: FinitePaletteStructuralEvidenceRequest,
    ) -> tuple[CalibratedPortfolioOptionEvidence, ...]:
        self.__post_init__()
        if type(request) is not FinitePaletteStructuralEvidenceRequest:
            raise TypeError("request must be exact structural-evidence request")
        request.__post_init__()
        contract = request.contract
        locus_by_option = {
            option.option_id: _changed_path_key(contract, index)
            for index, option in enumerate(contract.options)
        }
        family_counts = Counter(option.family for option in contract.options)
        locus_counts = Counter(locus_by_option.values())
        minimum_family_count = min(family_counts.values())
        minimum_locus_count = min(locus_counts.values())
        phenotype_by_option = dict(request.option_phenotype_sha256s)
        known = set(request.known_phenotype_sha256s)
        results: list[CalibratedPortfolioOptionEvidence] = []
        for option in sorted(contract.options, key=lambda value: value.option_id):
            locus_key = locus_by_option[option.option_id]
            phenotype_sha256 = phenotype_by_option[option.option_id]
            archive_novelty = 0.0 if phenotype_sha256 in known else 1.0
            family_rarity = minimum_family_count / family_counts[option.family]
            locus_rarity = minimum_locus_count / locus_counts[locus_key]
            structural_coverage = float((family_rarity + locus_rarity) / 2.0)
            evidence_record = {
                "schema_version": 1,
                "policy": {
                    "policy_id": self.policy_id,
                    "policy_version": self.policy_version,
                    "definition_sha256": self.definition_sha256,
                },
                "eligibility_receipt_sha256": (
                    request.eligibility_receipt_sha256
                ),
                "frozen_archive_snapshot_sha256": (
                    request.frozen_archive_snapshot_sha256
                ),
                "finite_contract_sha256": contract.identity_sha256,
                "option_id": option.option_id,
                "option_identity_sha256": option.identity_sha256,
                "phenotype_identity_sha256": phenotype_sha256,
                "family": option.family,
                "locus_key": locus_key,
                "family_count": family_counts[option.family],
                "minimum_family_count": minimum_family_count,
                "locus_count": locus_counts[locus_key],
                "minimum_locus_count": minimum_locus_count,
                "archive_novelty_score_hex": archive_novelty.hex(),
                "structural_coverage_score_hex": structural_coverage.hex(),
                "objective_values_consulted": False,
            }
            receipt_sha256 = hashlib.sha256(
                _EVIDENCE_DOMAIN + _canonical_json(evidence_record)
            ).hexdigest()
            results.append(
                CalibratedPortfolioOptionEvidence(
                    option_id=option.option_id,
                    option_identity_sha256=option.identity_sha256,
                    locus_key=locus_key,
                    phenotype_identity_sha256=phenotype_sha256,
                    structural_evidence=SlateStructuralEvidence(
                        frozen_archive_snapshot_sha256=(
                            request.frozen_archive_snapshot_sha256
                        ),
                        evidence_receipt_sha256=receipt_sha256,
                        archive_novelty_score=archive_novelty,
                        structural_coverage_score=structural_coverage,
                    ),
                )
            )
        return tuple(results)


__all__ = [
    "FinitePaletteStructuralEvidencePolicy",
    "FinitePaletteStructuralEvidenceRequest",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
]

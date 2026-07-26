from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.domain.typed_json import FrozenJsonObject, freeze_json
from agent_evolve.policies.selection.finite_palette_evidence import (
    FinitePaletteStructuralEvidencePolicy,
    FinitePaletteStructuralEvidenceRequest,
)
from agent_evolve.ports.variation_catalog import bind_finite_variation_catalog
from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.finite_variation_catalog import (
    BoilsFiniteVariationCatalog,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _parent() -> FrozenJsonObject:
    value = freeze_json({"sequence": list(DEFAULT_ACTION_SEQUENCE)})
    assert type(value) is FrozenJsonObject
    return value


def _request() -> FinitePaletteStructuralEvidenceRequest:
    contract = bind_finite_variation_catalog(
        BoilsFiniteVariationCatalog(),
        _parent(),
    )
    return FinitePaletteStructuralEvidenceRequest(
        contract=contract,
        option_phenotype_sha256s=tuple(
            sorted(
                (
                    option.option_id,
                    option.child_configuration_sha256,
                )
                for option in contract.options
            )
        ),
        known_phenotype_sha256s=(),
        eligibility_receipt_sha256=_sha("eligibility"),
        frozen_archive_snapshot_sha256=_sha("archive"),
    )


def test_projects_stable_objective_blind_evidence_for_complete_real_palette() -> None:
    request = _request()
    policy = FinitePaletteStructuralEvidencePolicy()

    first = policy.project(request)
    replay = policy.project(request)

    assert first == replay
    assert len(first) == 200
    assert tuple(value.option_id for value in first) == tuple(
        sorted(value.option_id for value in first)
    )
    assert len({value.locus_key for value in first}) == 20
    assert all(
        value.structural_evidence.frozen_archive_snapshot_sha256 == _sha("archive")
        for value in first
    )
    assert all(
        value.structural_evidence.archive_novelty_score == 1.0 for value in first
    )
    assert all(
        0.0 <= value.structural_evidence.structural_coverage_score <= 1.0
        for value in first
    )
    assert len(
        {value.structural_evidence.evidence_receipt_sha256 for value in first}
    ) == len(first)


def test_known_phenotype_changes_only_binary_novelty_not_identity_or_locus() -> None:
    request = _request()
    original = FinitePaletteStructuralEvidencePolicy().project(request)
    target = request.option_phenotype_sha256s[0]
    known_request = replace(request, known_phenotype_sha256s=(target[1],))

    observed = FinitePaletteStructuralEvidencePolicy().project(known_request)

    by_id = {value.option_id: value for value in observed}
    assert by_id[target[0]].structural_evidence.archive_novelty_score == 0.0
    assert sum(
        value.structural_evidence.archive_novelty_score == 0.0 for value in observed
    ) == 1
    assert tuple(value.locus_key for value in observed) == tuple(
        value.locus_key for value in original
    )
    assert tuple(value.option_identity_sha256 for value in observed) == tuple(
        value.option_identity_sha256 for value in original
    )


def test_request_rejects_partial_or_foreign_phenotype_bindings() -> None:
    request = _request()
    with pytest.raises(ValueError, match="cover the exact finite contract"):
        replace(
            request,
            option_phenotype_sha256s=request.option_phenotype_sha256s[:-1],
        )
    first_option, first_phenotype = request.option_phenotype_sha256s[0]
    with pytest.raises(ValueError, match="cannot repeat an option"):
        replace(
            request,
            option_phenotype_sha256s=tuple(
                sorted(
                    (
                        (first_option, first_phenotype),
                        (first_option, _sha("foreign")),
                        *request.option_phenotype_sha256s[1:],
                    )
                )
            ),
        )

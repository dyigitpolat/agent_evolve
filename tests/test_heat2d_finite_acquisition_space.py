from __future__ import annotations

import hashlib

from agent_evolve.domain.typed_json import (
    FrozenJsonObject,
    freeze_json,
    typed_json_sha256,
)
from agent_evolve.ports.finite_acquisition_space import (
    FiniteAcquisitionSpaceRequest,
    validate_finite_acquisition_space_candidates,
    validate_finite_acquisition_space_identity,
)
from examples.benchmarks.heat2d_constructive.candidate import seed_layouts
from examples.benchmarks.heat2d_constructive.finite_acquisition_space import (
    Heat2DFiniteAcquisitionSpace,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (
    LOCUS_GRIDS,
    Heat2DFiniteVariationCatalog,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _request(seed: int) -> FiniteAcquisitionSpaceRequest:
    observed = tuple(
        sorted(
            (
                freeze_json(value.model_dump(mode="python"))
                for value in seed_layouts()
            ),
            key=typed_json_sha256,
        )
    )
    assert all(type(value) is FrozenJsonObject for value in observed)
    return FiniteAcquisitionSpaceRequest(
        campaign_scope_sha256=_sha("heat-space-test"),
        cutoff_index=2,
        pool_size=32,
        seed=seed,
        observed_configurations=observed,  # type: ignore[arg-type]
        excluded_configuration_sha256s=tuple(
            sorted(typed_json_sha256(value) for value in observed)
        ),
    )


def test_heat_acquisition_space_is_exact_deterministic_and_outcome_blind():
    space = Heat2DFiniteAcquisitionSpace()
    assert validate_finite_acquisition_space_identity(space)[0] == (
        "heat2d_constructive_finite_acquisition"
    )
    request = _request(17)
    first = space.candidates(request)
    second = space.candidates(request)
    validate_finite_acquisition_space_candidates(
        request=request,
        candidates=first,
    )

    assert first == second
    assert len(first) == 32
    assert len({typed_json_sha256(value) for value in first}) == 32
    assert all(
        len(space.features(value)) == 1 + len(LOCUS_GRIDS) for value in first
    )
    assert all(
        0.0 <= coordinate <= 1.0
        for value in first
        for coordinate in space.features(value)
    )
    assert space.candidates(_request(18)) != first


def test_heat_space_reserves_prior_neighborhood_support():
    space = Heat2DFiniteAcquisitionSpace()
    request = _request(23)
    candidates = space.candidates(request)
    candidate_ids = {typed_json_sha256(value) for value in candidates}
    local_ids: set[str] = set()
    catalog = Heat2DFiniteVariationCatalog()
    for observed in request.observed_configurations:
        local_ids.update(
            option.child_configuration_sha256
            for option in catalog.options(observed)
        )

    assert len(candidate_ids & local_ids) == request.pool_size // 4

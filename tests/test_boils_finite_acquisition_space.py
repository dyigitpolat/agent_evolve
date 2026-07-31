from __future__ import annotations

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

from examples.benchmarks.boils_abc.actions import DEFAULT_ACTION_SEQUENCE
from examples.benchmarks.boils_abc.finite_acquisition_space import (
    BoilsFiniteAcquisitionSpace,
)


def _seed() -> FrozenJsonObject:
    value = freeze_json({"sequence": list(DEFAULT_ACTION_SEQUENCE)})
    assert type(value) is FrozenJsonObject
    return value


def _request(seed: int = 17) -> FiniteAcquisitionSpaceRequest:
    observed = _seed()
    return FiniteAcquisitionSpaceRequest(
        campaign_scope_sha256="1" * 64,
        cutoff_index=2,
        pool_size=64,
        seed=seed,
        observed_configurations=(observed,),
        excluded_configuration_sha256s=(typed_json_sha256(observed),),
    )


def test_boils_space_is_deterministic_legal_and_excludes_observations() -> None:
    space = BoilsFiniteAcquisitionSpace()
    assert validate_finite_acquisition_space_identity(space)[0] == (
        "boils_abc_finite_acquisition"
    )
    request = _request()
    first = space.candidates(request)
    second = space.candidates(request)
    assert first == second
    assert len(first) == 64
    assert typed_json_sha256(_seed()) not in {
        typed_json_sha256(value) for value in first
    }
    assert all(len(space.features(value)) == 20 for value in first)
    validate_finite_acquisition_space_candidates(
        request=request,
        candidates=first,
    )


def test_boils_space_seed_changes_global_reservoir() -> None:
    space = BoilsFiniteAcquisitionSpace()
    assert space.candidates(_request(17)) != space.candidates(_request(18))

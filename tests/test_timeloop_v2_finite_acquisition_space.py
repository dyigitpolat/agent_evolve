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

from examples.benchmarks.timeloop_codesign.v2.candidate import DEFAULT_CANDIDATE
from examples.benchmarks.timeloop_codesign.v2.finite_acquisition_space import (
    TimeloopV2FiniteAcquisitionSpace,
)
from examples.benchmarks.timeloop_codesign.v2.frozen_panels import (
    frozen_network_panel,
)


def _seed() -> FrozenJsonObject:
    value = freeze_json(DEFAULT_CANDIDATE)
    assert type(value) is FrozenJsonObject
    return value


def _request(seed: int = 17) -> FiniteAcquisitionSpaceRequest:
    observed = _seed()
    return FiniteAcquisitionSpaceRequest(
        campaign_scope_sha256="2" * 64,
        cutoff_index=2,
        pool_size=64,
        seed=seed,
        observed_configurations=(observed,),
        excluded_configuration_sha256s=(typed_json_sha256(observed),),
    )


def test_timeloop_space_is_deterministic_legal_and_excludes_observations() -> None:
    space = TimeloopV2FiniteAcquisitionSpace(frozen_network_panel("resnet50"))
    assert validate_finite_acquisition_space_identity(space)[0] == (
        "timeloop_codesign_v2_finite_acquisition"
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


def test_timeloop_space_seed_changes_global_reservoir() -> None:
    space = TimeloopV2FiniteAcquisitionSpace(frozen_network_panel("resnet50"))
    assert space.candidates(_request(17)) != space.candidates(_request(18))

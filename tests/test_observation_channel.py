"""The observation channel: carried, sealed, and never scored.

An evaluation unit that aggregates sub-problems publishes the parts and then
discards them, because the only typed channel out of an adapter was the
aggregate objective vector.  `observations` is that channel.  These tests pin
the three properties that make it safe to add:

  * every existing caller resolves identically -- the field defaults to empty
    and `evidence_sha256` is byte-identical for every payload that does not use
    it, so sealed campaign journals and prompt-shape commitments still verify;
  * it reaches the engine record, so a consumer can read it;
  * nothing scores it -- the reward path consumes `objectives` alone and the
    objective vector is untouched, so widening the observation cannot move the
    target.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_evolve.agentic import PhenotypeIdentity
from agent_evolve.application.detailed_evaluation import (
    DetailedEvaluation,
    DetailedEvaluationPayload,
    EvaluationTimings,
    EvaluatorIdentity,
    normalize_detailed_payload,
)
from agent_evolve.core.problem import ObjectiveSpec
from agent_evolve.domain.outcome import (
    FailureCategory,
    FailureCode,
    FailureRecord,
)

_EVALUATOR = EvaluatorIdentity(
    evaluator_id="unit_evaluator",
    evaluator_version=1,
    evaluator_context_sha256="c" * 64,
)
_PHENOTYPE = PhenotypeIdentity(
    policy_id="typed_configuration_phenotype",
    policy_version=1,
    value_sha256="a" * 64,
)
_OBJECTIVES = (("energy", 2.0), ("latency", 5.0))
_OBSERVATIONS = (
    ("medoid_0_energy", 0.25),
    ("medoid_1_energy", 0.75),
    ("medoid_0_latency", 1.5),
)


def _payload(**overrides) -> DetailedEvaluationPayload:
    fields = dict(
        failure=None,
        objectives=_OBJECTIVES,
        violations=(),
        checks=(),
        receipt=None,
        evaluator=_EVALUATOR,
    )
    fields.update(overrides)
    return DetailedEvaluationPayload(**fields)


def _sealed(payload: DetailedEvaluationPayload) -> DetailedEvaluation:
    return DetailedEvaluation(
        phenotype=_PHENOTYPE,
        payload=payload,
        timings=EvaluationTimings(total_wall_seconds=1.0),
    )


def test_observations_default_to_empty_so_existing_callers_are_unchanged() -> None:
    payload = _payload()
    assert payload.observations == ()
    assert _sealed(payload).observations == ()


def test_empty_observations_leave_the_seal_byte_identical() -> None:
    """The load-bearing backward-compatibility property.

    `evidence_sha256` is published: sealed campaign journals carry it, and the
    prompt-shape commitment binds it as `parent_evidence_sha256s` and
    `common_ancestor_evidence_sha256`.  An always-present key would re-hash
    every evaluation ever recorded.
    """
    sealed = _sealed(_payload())
    assert "observations" not in sealed._identity_record()
    # the digest of a payload with an explicitly-empty tuple must equal the
    # digest of one that never mentioned observations at all
    assert sealed.evidence_sha256 == _sealed(_payload(observations=())).evidence_sha256


def test_non_empty_observations_are_sealed_rather_than_merely_reported() -> None:
    bare = _sealed(_payload())
    carrying = _sealed(_payload(observations=_OBSERVATIONS))
    assert "observations" in carrying._identity_record()
    assert carrying.evidence_sha256 != bare.evidence_sha256


def test_observations_reach_the_engine_record() -> None:
    record = _sealed(_payload(observations=_OBSERVATIONS)).to_record()
    assert record["observations"] == {
        "medoid_0_energy": 0.25,
        "medoid_1_energy": 0.75,
        "medoid_0_latency": 1.5,
    }


def test_the_reward_vector_is_untouched_by_the_observation_channel() -> None:
    """Excluded from reward: the objective vector is identical either way."""
    bare = _sealed(_payload())
    carrying = _sealed(_payload(observations=_OBSERVATIONS))
    assert bare.objectives == carrying.objectives == _OBJECTIVES
    assert bare.to_record()["objectives"] == carrying.to_record()["objectives"]


def test_normalization_preserves_observations_and_reorders_only_objectives() -> None:
    specs = (ObjectiveSpec("latency", "min"), ObjectiveSpec("energy", "min"))
    payload = _payload(observations=_OBSERVATIONS)
    normalized = normalize_detailed_payload(payload, specs)
    assert tuple(name for name, _ in normalized.objectives) == ("latency", "energy")
    assert normalized.observations == _OBSERVATIONS


def test_dataclass_replace_carries_observations() -> None:
    payload = _payload(observations=_OBSERVATIONS)
    assert replace(payload, violations=()).observations == _OBSERVATIONS


def test_observations_may_not_shadow_an_objective_name() -> None:
    with pytest.raises(ValueError, match="shadow objective names"):
        _payload(observations=(("energy", 1.0),))


def test_observation_names_must_be_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        _payload(observations=(("m", 1.0), ("m", 2.0)))


def test_observation_values_must_be_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        _payload(observations=(("m", float("inf")),))


def test_observations_need_not_be_sorted() -> None:
    """Sub-problem order is meaningful; medoid 0, 1, 2 is not alphabetical."""
    payload = _payload(observations=(("z_first", 1.0), ("a_second", 2.0)))
    assert payload.observations == (("z_first", 1.0), ("a_second", 2.0))


def test_a_failed_evaluation_cannot_carry_observations() -> None:
    failure = FailureRecord(
        category=FailureCategory.CANDIDATE,
        code=FailureCode.SCHEMA_INVALID,
        message="bad",
        retryable=False,
        exception_type="ValueError",
    )
    with pytest.raises(ValueError, match="cannot carry projections"):
        _payload(failure=failure, objectives=(), observations=_OBSERVATIONS)

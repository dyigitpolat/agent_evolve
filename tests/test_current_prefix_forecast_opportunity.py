from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agent_evolve.application.agentic_evolution import EvolutionCandidate
from agent_evolve.application.current_prefix_forecast_opportunity import (
    CurrentPrefixForecastOpportunityPolicy,
)
from agent_evolve.application.forecast_geometry_portfolio import (
    ForecastGeometryScenario,
    MaterializedForecastGeometryBatch,
    MaterializedForecastGeometryMember,
)
from agent_evolve.domain.ids import CandidateId
from agent_evolve.domain.lineage import CandidateOccurrence
from agent_evolve.domain.typed_json import freeze_json, typed_json_sha256


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _candidate(
    candidate_id: str,
    *,
    x: float,
    y: float,
    generation: int,
) -> EvolutionCandidate:
    configuration = freeze_json({"candidate_id": candidate_id})
    return EvolutionCandidate(
        occurrence=CandidateOccurrence(
            candidate_id=CandidateId(candidate_id),
            configuration_hash=typed_json_sha256(configuration),
            configuration_artifact_hash=_sha(f"artifact:{candidate_id}"),
            proposal_sequence=generation + 1,
        ),
        configuration=configuration,
        objectives=(("x", x), ("y", y)),
        valid=True,
        generation=generation,
        label=candidate_id,
    )


class _CoverageUtility:
    utility_id = "test_prefix_coverage"
    utility_version = 1
    definition_sha256 = _sha("test-prefix-coverage")

    @staticmethod
    def _value(points) -> float:
        values = tuple(points)
        if not values:
            return 0.0
        return max(float(value["x"]) for value in values) + max(
            float(value["y"]) for value in values
        )

    def utility(self, candidates):
        return self._value(
            value.objective_map
            for value in candidates
            if (
                value.valid
                and value.operator_compliant
                and value.evidence_compliant
            )
        )

    def marginal_utility(self, candidates, objective_point):
        return self.portfolio_marginal_utility(
            candidates,
            (objective_point,),
        )

    def portfolio_marginal_utility(self, candidates, objective_points):
        before = self.utility(candidates)
        admitted = [
            value.objective_map
            for value in candidates
            if (
                value.valid
                and value.operator_compliant
                and value.evidence_compliant
            )
        ]
        return self._value((*admitted, *objective_points)) - before


def _member(
    action_id: str,
    *,
    x: float,
    y: float,
    reliability: float = 1.0,
    phenotype: str | None = None,
) -> MaterializedForecastGeometryMember:
    return MaterializedForecastGeometryMember(
        action_sha256=_sha(f"action:{action_id}"),
        phenotype_identity_sha256=_sha(
            f"phenotype:{action_id}" if phenotype is None else phenotype
        ),
        reliability=reliability,
        scenarios=tuple(
            sorted(
                (
                    ForecastGeometryScenario(
                        scenario_id="adverse",
                        objective_point=(
                            ("x", 0.8 * x),
                            ("y", 0.8 * y),
                        ),
                    ),
                    ForecastGeometryScenario(
                        scenario_id="favorable",
                        objective_point=(
                            ("x", 1.2 * x),
                            ("y", 1.2 * y),
                        ),
                    ),
                    ForecastGeometryScenario(
                        scenario_id="median",
                        objective_point=(("x", x), ("y", y)),
                    ),
                ),
                key=lambda value: value.scenario_id,
            )
        ),
        source_evidence_sha256=_sha(f"evidence:{action_id}"),
    )


def _geometry(
    members: tuple[MaterializedForecastGeometryMember, ...],
) -> MaterializedForecastGeometryBatch:
    return MaterializedForecastGeometryBatch(
        projection_id="test_prefix_geometry",
        projection_version=1,
        projection_definition_sha256=_sha("test-prefix-geometry"),
        residual_request_sha256=_sha("test-prefix-request"),
        proposal_sha256s=(_sha("test-prefix-proposal"),),
        members=tuple(
            sorted(members, key=lambda value: value.action_sha256)
        ),
        candidate_outcomes_observed=False,
        evidence=freeze_json(
            {
                "candidate_outcomes_observed": False,
                "test_geometry": True,
            }
        ),
    )


def test_real_prefix_suppresses_redundant_forecast_and_exposes_complement() -> None:
    a = _member("a", x=10.0, y=0.0)
    b = _member("b", x=9.0, y=0.0)
    c = _member("c", x=0.0, y=8.0)
    geometry = _geometry((a, b, c))
    seed = _candidate("candidate_seed", x=0.0, y=0.0, generation=0)
    policy = CurrentPrefixForecastOpportunityPolicy(
        archive_utility=_CoverageUtility(),
        risk_aversion=0.0,
    )

    initial = policy.rank(
        prior_candidates=(seed,),
        current_prefix_candidates=(),
        geometry=geometry,
    )
    assert initial.recommended_action_sha256s == (a.action_sha256,)
    initial_by_action = {
        value.action_sha256: value for value in initial.scores
    }
    assert initial_by_action[a.action_sha256].central_gain == 10.0
    assert initial_by_action[c.action_sha256].central_gain == 8.0
    assert initial.current_prefix_outcomes_observed is False

    observed_a = _candidate(
        "candidate_observed_a",
        x=10.0,
        y=0.0,
        generation=1,
    )
    continued = policy.rank(
        prior_candidates=(seed,),
        current_prefix_candidates=(observed_a,),
        geometry=geometry,
        consumed_action_sha256s=(a.action_sha256,),
    )
    assert continued.recommended_action_sha256s == (c.action_sha256,)
    continued_by_action = {
        value.action_sha256: value for value in continued.scores
    }
    assert continued_by_action[b.action_sha256].central_gain == 0.0
    assert continued_by_action[b.action_sha256].abstained is True
    assert continued_by_action[c.action_sha256].central_gain == 8.0
    assert continued.current_prefix_outcomes_observed is True
    assert continued.eligible_candidate_outcomes_observed is False
    assert initial.archive_sha256 != continued.archive_sha256


def test_recommendations_are_unique_by_phenotype_and_may_abstain() -> None:
    shared = "shared-phenotype"
    a = _member("a", x=10.0, y=0.0, phenotype=shared)
    b = _member("b", x=9.0, y=0.0, phenotype=shared)
    c = _member("c", x=0.0, y=8.0)
    weak = _member("weak", x=20.0, y=20.0, reliability=0.1)
    geometry = _geometry((a, b, c, weak))
    policy = CurrentPrefixForecastOpportunityPolicy(
        archive_utility=_CoverageUtility(),
        risk_aversion=0.0,
        minimum_reliability=0.5,
    )
    ranking = policy.rank(
        prior_candidates=(
            _candidate("candidate_seed", x=0.0, y=0.0, generation=0),
        ),
        current_prefix_candidates=(),
        geometry=geometry,
        recommendation_count=3,
    )
    assert set(ranking.recommended_action_sha256s) == {
        a.action_sha256,
        c.action_sha256,
    }
    weak_score = next(
        value
        for value in ranking.scores
        if value.action_sha256 == weak.action_sha256
    )
    assert weak_score.abstained is True
    assert weak_score.abstention_reason == "below_reliability_floor"
    assert ranking.to_record(include_scores=True)["ranking_sha256"] == (
        ranking.ranking_sha256
    )


def test_market_and_receipt_integrity_fail_closed() -> None:
    member = _member("a", x=1.0, y=1.0)
    geometry = _geometry((member,))
    policy = CurrentPrefixForecastOpportunityPolicy(
        archive_utility=_CoverageUtility(),
    )
    seed = _candidate("candidate_seed", x=0.0, y=0.0, generation=0)

    with pytest.raises(ValueError, match="escape"):
        policy.rank(
            prior_candidates=(seed,),
            current_prefix_candidates=(),
            geometry=geometry,
            consumed_action_sha256s=(_sha("foreign-action"),),
        )

    ranking = policy.rank(
        prior_candidates=(seed,),
        current_prefix_candidates=(),
        geometry=geometry,
        recommendation_count=0,
    )
    assert ranking.recommended_action_sha256s == ()
    with pytest.raises(ValueError, match="outcomes must remain hidden"):
        replace(
            ranking,
            eligible_candidate_outcomes_observed=True,
        )


def test_consumed_phenotype_is_removed_from_default_and_explicit_markets() -> None:
    first = _member("first", x=1.0, y=0.0, phenotype="shared")
    duplicate = _member("duplicate", x=2.0, y=0.0, phenotype="shared")
    distinct = _member("distinct", x=0.0, y=1.0)
    geometry = _geometry((first, duplicate, distinct))
    policy = CurrentPrefixForecastOpportunityPolicy(
        archive_utility=_CoverageUtility(),
        risk_aversion=0.0,
    )
    seed = _candidate("candidate_seed", x=0.0, y=0.0, generation=0)
    ranking = policy.rank(
        prior_candidates=(seed,),
        current_prefix_candidates=(
            _candidate("candidate_first", x=1.0, y=0.0, generation=1),
        ),
        geometry=geometry,
        consumed_action_sha256s=(first.action_sha256,),
    )
    assert ranking.eligible_action_sha256s == (distinct.action_sha256,)

    with pytest.raises(ValueError, match="consumed phenotype"):
        policy.rank(
            prior_candidates=(seed,),
            current_prefix_candidates=(),
            geometry=geometry,
            consumed_action_sha256s=(first.action_sha256,),
            eligible_action_sha256s=tuple(
                sorted(
                    (
                        distinct.action_sha256,
                        duplicate.action_sha256,
                    )
                )
            ),
        )

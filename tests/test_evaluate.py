"""evaluate_batch: validate_detailed / validate / evaluate failure routing."""

from __future__ import annotations

import math

import pytest

from agent_evolve.core.problem import ObjectiveSpec, ProblemContractError, ValidationOutcome
from agent_evolve.session.evaluate import evaluate_batch

OBJS = [ObjectiveSpec("score", "max"), ObjectiveSpec("penalty", "min")]


class _Detailed:
    objectives = OBJS

    def validate_detailed(self, config):
        if config.get("a", 0) > 10:
            return ValidationOutcome(False, "constraint", "a too large")
        return ValidationOutcome(True)

    def evaluate(self, config):
        return {"score": float(config["a"]), "penalty": float(config.get("b", 0))}


class _Legacy:
    objectives = OBJS

    def validate(self, config):
        if "a" not in config:
            raise ValueError("missing 'a'")
        return True

    def evaluate(self, config):
        return {"score": float(config["a"]), "penalty": 0.0}


def test_validate_detailed_failure_phase():
    valid, failed, ordered = evaluate_batch(_Detailed(), [{"a": 3, "b": 1}, {"a": 20}], OBJS)
    assert len(valid) == 1 and len(failed) == 1
    assert failed[0].failure_phase == "constraint"
    assert failed[0].evaluation_attempted is False
    assert valid[0].evaluation_attempted is True
    assert "too large" in failed[0].error_message
    assert len(ordered) == 2


def test_legacy_validate_raises_become_feedback():
    valid, failed, _ = evaluate_batch(_Legacy(), [{"a": 5}, {"b": 2}], OBJS)
    assert len(valid) == 1 and len(failed) == 1
    assert failed[0].failure_phase == "validation"
    assert "missing 'a'" in failed[0].error_message


def test_evaluate_exception_phase():
    class _Boom:
        objectives = OBJS

        def evaluate(self, config):
            raise ValueError("boom")

    valid, failed, _ = evaluate_batch(_Boom(), [{"a": 1}], OBJS)
    assert not valid and failed[0].failure_phase == "evaluation"
    assert failed[0].evaluation_attempted is True


def test_infrastructure_error_from_evaluate_propagates():
    class _Disconnected:
        objectives = OBJS

        def evaluate(self, config):
            raise ConnectionError("simulator container unavailable")

    with pytest.raises(ConnectionError, match="container unavailable"):
        evaluate_batch(_Disconnected(), [{"a": 1}], OBJS)


def test_programming_error_from_legacy_validate_propagates():
    class _BrokenValidator:
        objectives = OBJS

        def validate(self, config):
            raise RuntimeError("validator bug")

        def evaluate(self, config):
            return {"score": 1.0, "penalty": 0.0}

    with pytest.raises(RuntimeError, match="validator bug"):
        evaluate_batch(_BrokenValidator(), [{"a": 1}], OBJS)


@pytest.mark.parametrize(
    "objectives",
    [
        {"score": 1.0},
        {"score": 1.0, "penalty": 0.0, "typo": 2.0},
        {"score": True, "penalty": 0.0},
        {"score": "1", "penalty": 0.0},
        {"score": math.nan, "penalty": 0.0},
        {"score": math.inf, "penalty": 0.0},
        {"score": -math.inf, "penalty": 0.0},
        [1.0, 0.0],
    ],
)
def test_invalid_objective_contract_aborts(objectives):
    class _BadObjectives:
        def evaluate(self, config):
            return objectives

    with pytest.raises(ProblemContractError):
        evaluate_batch(_BadObjectives(), [{"a": 1}], OBJS)


# --------------------------------------------------- refusals are charged ONCE

class _RefusingVenue:
    """A venue that refuses a large share of what is proposed (the EDA case:

    53% of proposed configurations came back refused). A refusal RUNS the
    evaluator and is charged -- and before W9's fix it was charged AGAIN every
    time the sampler re-proposed the same configuration, because only
    successes were recorded in the cache.
    """

    objectives = OBJS

    def __init__(self):
        self.evaluator_runs = 0

    def evaluate(self, config):
        self.evaluator_runs += 1
        if config["a"] % 2:
            raise ValueError(f"refused: a={config['a']} violates the DRC deck")
        return {"score": float(config["a"]), "penalty": 0.0}


def test_a_reproposed_refusal_is_replayed_and_never_charged_twice():
    from agent_evolve.session.evaluate import EvaluationCache

    venue = _RefusingVenue()
    cache = EvaluationCache()
    # An EDA-shaped proposal stream: about half the distinct configs refuse,
    # and the sampler re-proposes everything it has already tried.
    proposals = [{"a": i % 8} for i in range(32)]   # 8 distinct, 32 proposed
    valid, failed, ordered = evaluate_batch(venue, proposals, OBJS, cache=cache)

    assert venue.evaluator_runs == 8                # distinct artifacts, once each
    assert cache.misses == 8                        # ... and that is the whole bill
    assert cache.hits == 12                         # 4 successes replayed 3x
    assert cache.refusal_hits == 12                 # 4 refusals replayed 3x
    assert len(ordered) == 32

    # The replayed refusal carries the SAME feedback as the charged one and
    # does not claim an evaluation happened here.
    charged = [r for r in failed if r.evaluation_attempted]
    replayed = [r for r in failed if not r.evaluation_attempted]
    assert len(charged) == 4 and len(replayed) == 12
    assert {r.failure_phase for r in replayed} == {"evaluation"}
    assert {r.error_message for r in replayed} <= {r.error_message for r in charged}


def test_a_remembered_refusal_is_served_even_when_the_budget_is_spent():
    from agent_evolve.session.evaluate import EvaluationCache

    venue = _RefusingVenue()
    cache = EvaluationCache()
    cache.budget = 2
    evaluate_batch(venue, [{"a": 1}, {"a": 2}], OBJS, cache=cache)
    assert cache.exhausted()

    # Re-proposing the refused config is a REPLAY of its recorded outcome,
    # not a budget event: the caller learns why it failed, not that the
    # budget is spent, and nothing more is billed.
    _valid, failed, _ordered = evaluate_batch(venue, [{"a": 1}], OBJS, cache=cache)
    assert failed[0].failure_phase == "evaluation"
    assert "refused" in failed[0].error_message
    assert venue.evaluator_runs == 2
    assert cache.misses == 2

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

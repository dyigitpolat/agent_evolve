"""Model-authored surrogates: authorship is the model's, arbitration never is.

The authoring seam mirrors ``llm_prior``: whole replies, never repaired,
every rejection counted. The arbitration seam is the same validation gate
the rule surrogates face -- best-passing on held-out data screens, so an
authored surrogate is USED exactly when it out-predicts the rules on data it
never fitted, and a run whose authorship failed is a run the rules carry.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Sequence

from pydantic import BaseModel

from agent_evolve import optimize
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.policies.llm_surrogate import (
    AuthorTelemetry,
    author_surrogate,
    authored_surrogate_builder,
)
from agent_evolve.core.authored import authored_artifact
from agent_evolve.infrastructure.authored_runtime import AuthoredRuntime
from agent_evolve.policies.surrogate import validate_surrogate
from agent_evolve.session.screening import Screening

SPECS = [ObjectiveSpec("ones", "max"), ObjectiveSpec("zeros", "min")]

GOOD_SOURCE = (
    "def fit_predict(train_x, train_y, test_x):\n"
    "    # per-position average effect of a set bit, plus the grand mean\n"
    "    names = sorted({k for row in train_y for k in row})\n"
    "    grand = {n: sum(r[n] for r in train_y) / len(train_y) for n in names}\n"
    "    out = []\n"
    "    for x in test_x:\n"
    "        ones = float(sum(x['genome']))\n"
    "        out.append({'ones': ones, 'zeros': float(len(x['genome'])) - ones})\n"
    "    return out\n"
)

GOOD_REPLY = "Here is the surrogate.\n```python\n" + GOOD_SOURCE + "```\n"


def _author(reply, tel=None, attempts=1):
    return author_surrogate(
        lambda _p: reply, objectives=SPECS, schema_text="genome: 6 bits",
        attempts=attempts, telemetry=tel or AuthorTelemetry())


# --- authoring: accept whole or not at all -----------------------------------

def test_a_fenced_fit_predict_is_accepted_and_hashed():
    tel = AuthorTelemetry()
    artifact = _author(GOOD_REPLY, tel)
    assert artifact is not None
    assert artifact.kind == "surrogate" and artifact.authored_by == "llm"
    assert len(artifact.source_sha256) == 64
    assert tel.accepted == 1 and tel.calls == 1


def test_prose_without_a_code_block_is_counted_and_yields_nothing():
    tel = AuthorTelemetry()
    assert _author("I would fit a spline.", tel, attempts=2) is None
    assert tel.no_code_block == 2 and tel.calls == 2, "attempts not honoured"


def test_the_wrong_entry_point_is_refused():
    tel = AuthorTelemetry()
    reply = "```python\ndef predict(x):\n    return []\n```"
    assert _author(reply, tel) is None
    assert tel.wrong_entry_point == 1


def test_a_forbidden_import_is_refused_at_authoring_time():
    tel = AuthorTelemetry()
    reply = ("```python\nimport os\n"
             "def fit_predict(a, b, c):\n    return []\n```")
    assert _author(reply, tel) is None
    assert tel.forbidden_import == 1


def test_unparseable_source_is_counted():
    tel = AuthorTelemetry()
    assert _author("```python\ndef fit_predict(:\n```", tel) is None
    assert tel.unparseable == 1


def test_a_provider_error_is_counted_and_never_raised():
    tel = AuthorTelemetry()

    def boom(_p):
        raise RuntimeError("provider down")

    assert author_surrogate(boom, objectives=SPECS, schema_text="s",
                            telemetry=tel) is None
    assert tel.errors >= 1


# --- the authored builder passes the same gate -------------------------------

LEARNABLE = [
    ({"genome": list(row)},
     {"ones": float(sum(row)), "zeros": float(len(row) - sum(row))})
    for row in ([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 1, 0],
                [0, 0, 0, 1, 1, 1], [1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 1, 1])
]


def test_an_authored_surrogate_runs_out_of_process_and_validates():
    artifact = authored_artifact("surrogate", GOOD_SOURCE,
                                 name="exact", authored_by="llm")
    builder = authored_surrogate_builder(artifact, AuthoredRuntime())
    verdict = validate_surrogate(builder, LEARNABLE, SPECS, seed=4)
    assert verdict.passed, verdict


def test_best_passing_arbitration_prefers_the_better_predictor():
    # The authored source above is EXACT; the biased rule below passes the
    # gate but with worse error. The screen must choose by measurement.
    def biased(evaluated, specs):
        from agent_evolve.policies.surrogate import additive_surrogate
        full = additive_surrogate(evaluated, specs)

        def predict(pool):
            rows = full(pool)
            if rows is None:
                return None
            return [{k: v + 0.4 for k, v in row.items()} for row in rows]

        return predict

    artifact = authored_artifact("surrogate", GOOD_SOURCE,
                                 name="exact", authored_by="llm")
    screening = Screening(builders=(
        ("biased", "rule", biased),
        ("llm:exact", "llm",
         authored_surrogate_builder(artifact, AuthoredRuntime())),
    ))
    assert screening.refresh(LEARNABLE, SPECS, seed=5)
    assert screening.authored_by == "llm", (
        "the exact authored surrogate lost arbitration to a biased rule"
    )
    assert screening.telemetry.chosen_llm == 1


# --- end to end through optimize() ------------------------------------------

class _Candidate(BaseModel):
    genome: list[Literal[0, 1]]


class _Problem:
    candidate_model = _Candidate
    objectives = tuple(SPECS)

    def __init__(self) -> None:
        self.calls = 0

    def seeds(self) -> Sequence[Dict[str, Any]]:
        return ({"genome": [0, 0, 0, 0, 0, 0]},)

    def validate(self, config) -> ValidationOutcome:
        return ValidationOutcome(ok=True)

    def materialize(self, config) -> Any:
        return tuple(config["genome"])

    def evaluate(self, artifact) -> Mapping[str, float]:
        self.calls += 1
        ones = float(sum(artifact))
        return {"ones": ones, "zeros": float(len(artifact)) - ones}


def test_surrogate_llm_end_to_end_with_a_canned_author(monkeypatch):
    def _canned_completion_for(model, settings=None, **kwargs):
        return lambda prompt: GOOD_REPLY

    monkeypatch.setattr(
        "agent_evolve.integrations.completion.completion_for",
        _canned_completion_for,
    )
    problem = _Problem()
    result = optimize(problem, budget=24, seed=11, proposer="llm",
                      authorship="surrogate-llm")
    assert problem.calls <= 24 and result.evaluations <= 24
    mechanisms = {m.mechanism: m for m in result.telemetry.mechanisms}
    assert "surrogate_author" in mechanisms
    assert mechanisms["surrogate_author"].counters["accepted"] == 1
    screen = mechanisms.get("surrogate_screen")
    assert screen is not None and screen.counters["chosen_llm"] >= 1, (
        "an exact authored surrogate never won arbitration"
    )


def test_surrogate_llm_without_a_credential_falls_back_out_loud():
    said: list[str] = []
    result = optimize(_Problem(), budget=16, seed=12,
                      authorship="surrogate-llm", on_progress=said.append)
    assert any("rule surrogates" in m for m in said)
    assert result.telemetry.virtual_evaluations > 0, (
        "the rule surrogates should still carry the screen"
    )

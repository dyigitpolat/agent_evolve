"""The seal has to survive the ways a generative record can lie.

Each test names a specific way an unaudited generative loop would let a false
claim through, and shows the seal refusing it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from agent_evolve.application.generative_proposal_journal import (
    candidate_schema_sha256,
    read_generative_journal,
    verify_generative_journal,
    write_generative_journal,
)
from agent_evolve.core.problem import ObjectiveSpec, ValidationOutcome
from agent_evolve.domain.generative_emission import (
    GENESIS_CALL_SHA256,
    GenerativeEmission,
    GenerativeProposalCall,
    chain_sealed_calls,
    generative_prompt_sha256,
)
from agent_evolve.domain.typed_json import freeze_json
from agent_evolve.harness.base import HarnessContext, LLMConfig
from agent_evolve.harness.generative_seal import (
    SealedGenerativeHarness,
    SealedReplayDriftError,
)
from agent_evolve.proposal_mode import (
    CATALOGUE,
    GENERATIVE,
    ProposalSupport,
    build_generative_proposer,
    build_matched_null_proposer,
    describe_support,
    require_matched_support,
)


class Candidate(BaseModel):
    x: int = Field(ge=0, le=9)


class Problem:
    candidate_model = Candidate
    objectives = [ObjectiveSpec("y", "min")]

    def seeds(self):
        return ()

    def validate(self, config):
        if int(config["x"]) % 2:
            return ValidationOutcome(False, "constraint", "x must be even")
        return ValidationOutcome(True)

    def materialize(self, config):
        return dict(config)

    def evaluate(self, artifact):
        return {"y": float(artifact["x"])}


class ScriptedProposer:
    """A stand-in provider: returns what it was told to, and counts being asked."""

    id = "scripted"
    provides_insights = True

    def __init__(self, batches: List[List[Dict[str, Any]]]) -> None:
        self._batches = list(batches)
        self.calls = 0

    def bind(self, ctx: HarnessContext, cfg: LLMConfig) -> None:
        self._ctx, self._cfg = ctx, cfg

    def _next(self):
        self.calls += 1
        return self._batches.pop(0) if self._batches else [{"x": 0}]

    def generate_initial(self, n):
        return self._next()

    def regenerate(self, failed_str, n, ci, pi):
        return self._next()

    def generate_offspring(self, pareto_str, n, ci, pi):
        return self._next()

    def regenerate_offspring(self, failed_str, pareto_str, n, ci, pi):
        return self._next()

    def failure_insights(self, failed_str, n_failed):
        self.calls += 1
        return ["look at x"]

    def constraint_instruction(self, failed_str, previous=None):
        self.calls += 1
        return "keep x even"

    def performance_insights(self, stats_str, pareto_str, previous=None):
        self.calls += 1
        return "small x is better"


def _bind(harness, problem=None):
    problem = problem or Problem()
    harness.bind(
        HarnessContext(
            objectives=list(problem.objectives),
            search_space_desc="x in 0..9",
            candidate_model=problem.candidate_model,
        ),
        LLMConfig(model="openai/gpt-5.6-luna"),
    )
    return harness


# -- the emission record -------------------------------------------------


def test_a_rejected_emission_must_carry_the_reason_the_proposer_was_shown():
    with pytest.raises(ValueError, match="must carry the reason"):
        GenerativeEmission(configuration=freeze_json({"x": 1}), accepted=False)


def test_an_accepted_emission_cannot_also_carry_a_rejection():
    with pytest.raises(ValueError, match="no rejection reason"):
        GenerativeEmission(
            configuration=freeze_json({"x": 2}), accepted=True, rejection_reason="no"
        )


def test_identity_binds_the_configuration_not_a_label():
    a = GenerativeEmission(configuration=freeze_json({"x": 2}), accepted=True)
    b = GenerativeEmission(configuration=freeze_json({"x": 4}), accepted=True)
    assert a.identity_sha256 != b.identity_sha256


def test_a_call_that_returned_nothing_cannot_be_sealed_as_a_success():
    with pytest.raises(ValueError, match="at least one emission"):
        GenerativeProposalCall(
            call_ordinal=0,
            op="generate_offspring",
            requested_model="openai/gpt-5.6-luna",
            prompt_sha256=generative_prompt_sha256("q"),
            candidate_schema_sha256=candidate_schema_sha256(Candidate),
            emissions=(),
        )


# -- the chain -----------------------------------------------------------


def _call(ordinal, previous, x=2):
    return GenerativeProposalCall(
        call_ordinal=ordinal,
        op="generate_offspring",
        requested_model="openai/gpt-5.6-luna",
        prompt_sha256=generative_prompt_sha256(f"prompt {ordinal}"),
        candidate_schema_sha256=candidate_schema_sha256(Candidate),
        emissions=(GenerativeEmission(freeze_json({"x": x}), True),),
        previous_call_sha256=previous,
    )


def test_a_call_that_did_not_happen_cannot_be_spliced_into_the_chain():
    first = _call(0, GENESIS_CALL_SHA256)
    second = _call(1, first.identity_sha256)
    assert chain_sealed_calls((first, second))

    # The fabrication: a third call inserted between them, claiming the same
    # predecessor the real second call claims.
    forged = _call(1, first.identity_sha256, x=8)
    renumbered = GenerativeProposalCall(
        call_ordinal=2,
        op=second.op,
        requested_model=second.requested_model,
        prompt_sha256=second.prompt_sha256,
        candidate_schema_sha256=second.candidate_schema_sha256,
        emissions=second.emissions,
        previous_call_sha256=second.previous_call_sha256,
    )
    with pytest.raises(ValueError, match="does not follow"):
        chain_sealed_calls((first, forged, renumbered))


def test_a_chain_that_starts_mid_run_has_no_head():
    orphan = _call(0, "a" * 64)
    with pytest.raises(ValueError, match="does not follow"):
        chain_sealed_calls((orphan,))


# -- record then replay --------------------------------------------------


def test_replay_reproduces_the_run_without_the_delegate(tmp_path):
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}, {"x": 3}], [{"x": 4}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))

    first = recorder.generate_initial(2)
    recorder.performance_insights("stats", "pareto")
    second = recorder.generate_offspring("pareto", 1, "ci", "pi")

    journal = tmp_path / "calls.jsonl"
    terminal = write_generative_journal(journal, recorder.calls)
    assert terminal == recorder.terminal_sha256
    assert scripted.calls == 3

    sealed = read_generative_journal(journal)
    replayer = _bind(
        build_generative_proposer(
            problem, delegate=None, mode="replay", sealed_calls=sealed
        )
    )
    assert replayer.generate_initial(2) == first
    assert replayer.performance_insights("stats", "pareto") == "small x is better"
    assert replayer.generate_offspring("pareto", 1, "ci", "pi") == second
    assert replayer.terminal_sha256 == terminal


def test_the_seal_records_what_validate_said_not_what_the_model_hoped(tmp_path):
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}, {"x": 3}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    recorder.generate_initial(2)

    call = recorder.calls[0]
    assert [e.accepted for e in call.emissions] == [True, False]
    assert "x must be even" in call.emissions[1].rejection_reason

    journal = tmp_path / "calls.jsonl"
    write_generative_journal(journal, recorder.calls)
    summary = verify_generative_journal(journal)
    assert summary["emitted_configurations"] == 2
    assert summary["accepted_configurations"] == 1
    assert summary["requested_models"] == ["openai/gpt-5.6-luna"]


def test_a_drifted_prompt_is_an_error_not_a_live_call(tmp_path):
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    recorder.generate_offspring("pareto A", 1, "ci", "pi")
    sealed = recorder.calls

    replayer = _bind(
        build_generative_proposer(
            problem, delegate=None, mode="replay", sealed_calls=sealed
        )
    )
    with pytest.raises(SealedReplayDriftError, match="not the sealed"):
        replayer.generate_offspring("pareto B", 1, "ci", "pi")


def test_replay_refuses_when_the_feasibility_rule_moved_under_the_seal():
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    recorder.generate_initial(1)

    class Stricter(Problem):
        def validate(self, config):
            return ValidationOutcome(False, "constraint", "nothing is legal now")

    stricter = Stricter()
    replayer = _bind(
        build_generative_proposer(
            stricter, delegate=None, mode="replay", sealed_calls=recorder.calls
        ),
        stricter,
    )
    with pytest.raises(SealedReplayDriftError, match="feasibility rule changed"):
        replayer.generate_initial(1)


def test_replay_refuses_an_emission_drawn_from_a_different_schema():
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    recorder.generate_initial(1)

    class Wider(BaseModel):
        x: int = Field(ge=0, le=99)

    class WiderProblem(Problem):
        candidate_model = Wider

    wider = WiderProblem()
    replayer = _bind(
        build_generative_proposer(
            wider, delegate=None, mode="replay", sealed_calls=recorder.calls
        ),
        wider,
    )
    with pytest.raises(SealedReplayDriftError, match="different candidate schema"):
        replayer.generate_initial(1)


def test_replay_cannot_serve_more_calls_than_were_sealed():
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    recorder.generate_initial(1)

    replayer = _bind(
        build_generative_proposer(
            problem, delegate=None, mode="replay", sealed_calls=recorder.calls
        )
    )
    replayer.generate_initial(1)
    with pytest.raises(SealedReplayDriftError, match="journal is exhausted"):
        replayer.generate_initial(1)


def test_an_edited_journal_line_does_not_authenticate(tmp_path):
    problem = Problem()
    scripted = ScriptedProposer([[{"x": 2}]])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    recorder.generate_initial(1)
    journal = tmp_path / "calls.jsonl"
    write_generative_journal(journal, recorder.calls)

    text = journal.read_text(encoding="ascii").replace('"x":2', '"x":8')
    journal.write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match="does not authenticate"):
        verify_generative_journal(journal)


# -- the matched null ----------------------------------------------------


def test_the_null_samples_the_treatment_schema_and_says_so():
    problem = Problem()
    null = build_matched_null_proposer(problem, seed=7)
    _bind(null)
    drawn = null.generate_initial(64)
    assert all(0 <= c["x"] <= 9 for c in drawn)
    assert len({c["x"] for c in drawn}) > 1

    treatment = describe_support(problem, GENERATIVE)
    require_matched_support(treatment, describe_support(problem, GENERATIVE))


def test_comparing_a_generative_arm_against_a_catalogue_null_is_refused():
    problem = Problem()
    with pytest.raises(ValueError, match="support mismatch"):
        require_matched_support(
            describe_support(problem, GENERATIVE), ProposalSupport(mode=CATALOGUE)
        )


def test_comparing_across_two_schemas_is_refused():
    class Wider(BaseModel):
        x: int = Field(ge=0, le=99)

    class WiderProblem(Problem):
        candidate_model = Wider

    with pytest.raises(ValueError, match="different candidate schemas"):
        require_matched_support(
            describe_support(Problem(), GENERATIVE),
            describe_support(WiderProblem(), GENERATIVE),
        )


def test_a_problem_without_a_schema_cannot_run_a_generative_campaign():
    class Schemaless:
        candidate_model = None
        objectives = [ObjectiveSpec("y", "min")]

    with pytest.raises(ValueError, match="no declared support"):
        describe_support(Schemaless(), GENERATIVE)


# -- the loop actually runs through it -----------------------------------


def test_the_sealed_harness_drives_a_whole_optimization_and_replays_it(tmp_path):
    from agent_evolve.session.evaluate import EvaluationCache
    from agent_evolve.session.loop import LoopConfig, run_evolution_loop

    problem = Problem()
    batches = [[{"x": 8}, {"x": 6}], [{"x": 4}, {"x": 2}], [{"x": 0}, {"x": 6}]]

    def run(harness):
        cache = EvaluationCache()
        return run_evolution_loop(
            problem=problem,
            harness=harness,
            config=LoopConfig(
                pop_size=2,
                generations=2,
                candidates_per_batch=2,
                seed=3,
                evaluation_budget=6,
                evaluation_cache=cache,
                use_failure_insights=False,
            ),
        )

    scripted = ScriptedProposer([list(b) for b in batches])
    recorder = _bind(build_generative_proposer(problem, delegate=scripted))
    live = run(recorder)
    journal = tmp_path / "calls.jsonl"
    write_generative_journal(journal, recorder.calls)

    replayer = _bind(
        build_generative_proposer(
            problem,
            delegate=None,
            mode="replay",
            sealed_calls=read_generative_journal(journal),
        )
    )
    replayed = run(replayer)

    assert replayed.best.objectives == live.best.objectives
    assert replayed.evaluations == live.evaluations
    assert replayer.terminal_sha256 == recorder.terminal_sha256


def test_optimize_seals_the_run_when_asked(tmp_path):
    """The public entry point can produce a checkable journal in one argument."""

    from agent_evolve.api import optimize

    journal = tmp_path / "journal.jsonl"
    result = optimize(
        Problem(), budget=6, proposer="random", seed=11, seal=str(journal)
    )
    assert result.evaluations > 0

    summary = verify_generative_journal(journal)
    assert summary["proposal_calls"] >= 1
    assert summary["emitted_configurations"] >= summary["accepted_configurations"]
    assert summary["candidate_schema_sha256s"] == [candidate_schema_sha256(Candidate)]

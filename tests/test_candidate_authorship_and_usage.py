"""Product pins: a caller must be able to tell what the model produced and what it cost.

Both are answered from the result object alone. Both are counted rather than
declared, and both distinguish "measured zero" from "not recorded" -- the
distinction this project spent a week learning to keep.
"""

from __future__ import annotations

import pytest

from agent_evolve.core.results import Candidate, ProviderUsageSummary, SearchResult
from agent_evolve.session.loop import (
    AUTHORED_BY_CALLER_SEED,
    AUTHORED_BY_INITIAL_PROPOSAL,
    AUTHORED_BY_OFFSPRING_PROPOSAL,
    AUTHORING_CALLS,
    _candidate_metadata,
    _provider_usage,
)
from agent_evolve.core.formatting import CandidateResult


def _result(**kw):
    return CandidateResult(
        configuration=kw.get("configuration", {"a": 1}),
        objectives=kw.get("objectives", {"o": 1.0}),
        is_valid=True,
        evaluation_attempted=True,
    )


def test_authorship_vocabulary_is_closed():
    assert AUTHORING_CALLS == (
        AUTHORED_BY_CALLER_SEED,
        AUTHORED_BY_INITIAL_PROPOSAL,
        AUTHORED_BY_OFFSPRING_PROPOSAL,
    )


def test_metadata_records_the_authoring_call():
    meta = _candidate_metadata(
        _result(), generation=1, authored_by=AUTHORED_BY_INITIAL_PROPOSAL
    )
    assert meta["authored_by"] == AUTHORED_BY_INITIAL_PROPOSAL
    assert meta["generation"] == 1


def test_an_unnamed_authoring_call_is_rejected():
    """A new call site must state who authored, not inherit a default."""

    with pytest.raises(ValueError, match="authored_by must name an authoring call"):
        _candidate_metadata(_result(), generation=1, authored_by="somewhere")
    with pytest.raises(TypeError):
        _candidate_metadata(_result(), generation=1)  # no default to fall back on


def test_per_arm_counts_are_reportable_from_the_result():
    """The join is checkable, not asserted."""

    seed = Candidate({"a": 0}, {"o": 0.0}, {"authored_by": AUTHORED_BY_CALLER_SEED})
    first = Candidate({"a": 1}, {"o": 1.0}, {"authored_by": AUTHORED_BY_INITIAL_PROPOSAL})
    later = Candidate({"a": 2}, {"o": 2.0}, {"authored_by": AUTHORED_BY_OFFSPRING_PROPOSAL})
    result = SearchResult(objectives=[], best=first, all_candidates=[seed, first, later])
    assert result.candidates_by_author() == {
        AUTHORED_BY_CALLER_SEED: 1,
        AUTHORED_BY_INITIAL_PROPOSAL: 1,
        AUTHORED_BY_OFFSPRING_PROPOSAL: 1,
    }
    assert len(result.proposed_candidates()) == 2, "the caller's own seeds are not proposals"


def test_unlabelled_candidates_are_counted_as_unrecorded_not_as_a_guess():
    stray = Candidate({"a": 9}, {"o": 9.0}, {})
    result = SearchResult(objectives=[], best=stray, all_candidates=[stray])
    assert result.candidates_by_author() == {"unrecorded": 1}


def test_usage_counts_calls_and_reports_a_measured_zero():
    class _NoUsage:
        pass

    usage = _provider_usage(_NoUsage(), 0)
    assert usage.calls == 0 and usage.provider_free is True


def test_unreported_cost_is_none_and_not_zero():
    """None means 'this proposer does not report it'. Zero would be a claim."""

    class _NoUsage:
        pass

    usage = _provider_usage(_NoUsage(), 3)
    assert usage.calls == 3
    assert usage.cost_usd is None, "an unreported cost must not read as zero"


def test_an_unattributed_figure_is_unrepresentable():
    """The structural version: a zero nobody measured cannot be constructed."""

    with pytest.raises(ValueError, match="require reported_by"):
        ProviderUsageSummary(calls=0, cost_usd="0.00")
    with pytest.raises(ValueError, match="require reported_by"):
        ProviderUsageSummary(calls=3, input_tokens=0)
    # a genuinely measured zero is fine, because it names what measured it
    measured = ProviderUsageSummary(calls=0, cost_usd="0.00", reported_by="RandomHarness")
    assert measured.cost_is_known and measured.provider_free


def test_unreported_tokens_are_none_not_zero():
    """The bug this fix found: tokens defaulted to 0 when unreported."""

    class _NoUsage:
        pass

    usage = _provider_usage(_NoUsage(), 4)
    assert usage.input_tokens is None and usage.output_tokens is None
    assert usage.reported_by is None
    assert usage.cost_is_known is False


def test_a_reporting_harness_is_passed_through():
    class _Reports:
        def usage(self):
            return {
                "input_tokens": 1_200_000,
                "output_tokens": 700_000,
                "cost_usd": "0.54",
                "model": "openai/gpt-5.6-luna",
            }

    usage = _provider_usage(_Reports(), 7)
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (7, 1_200_000, 700_000)
    assert usage.cost_usd == "0.54" and usage.model == "openai/gpt-5.6-luna"
    assert usage.reported_by == "_Reports", "a figure must name what measured it"


def test_a_broken_harness_usage_does_not_lose_the_run():
    class _Raises:
        def usage(self):
            raise RuntimeError("harness defect")

    usage = _provider_usage(_Raises(), 2)
    assert usage.calls == 2 and usage.cost_usd is None

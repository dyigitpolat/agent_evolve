"""The corpus resolver, and the property that makes it safe to use.

corpus-free-by-construction: these build a miniature corpus in tmp_path, so
they mention every research signal there is and need none of them.

These run offline and need no corpus of their own -- they build one.
"""

from __future__ import annotations

import pytest

from examples.development.corpus_paths import corpus_path_or_none, resolve_corpus_path


@pytest.fixture
def corpus(tmp_path):
    """A miniature corpus with one live file and one that only exists archived."""
    root = tmp_path / "papers" / "agent_evolve_aaai_2027" / "research_artifacts"
    (root / "data").mkdir(parents=True)
    (root / "data" / "live.json").write_text('{"where": "live"}')
    (root / "archive" / "data").mkdir(parents=True)
    (root / "archive" / "data" / "moved.json").write_text('{"where": "archived"}')
    return root


def test_a_path_that_already_resolves_is_returned_untouched(corpus):
    """The fallback must never change an answer that was already correct."""
    live = corpus / "data" / "live.json"
    assert resolve_corpus_path(live) == live


def test_a_path_moved_by_the_split_resolves_to_the_archived_copy(corpus):
    moved = corpus / "data" / "moved.json"
    assert not moved.exists()
    assert resolve_corpus_path(moved) == corpus / "archive" / "data" / "moved.json"


def test_a_file_present_in_both_places_prefers_the_live_one(corpus):
    """The reason this is a fallback and not a repointed root.

    Repointing would move every path at once, so a file existing in both
    places would silently start resolving to the archived copy. It must not.
    """
    both = corpus / "data" / "both.json"
    both.write_text('{"where": "live"}')
    (corpus / "archive" / "data" / "both.json").write_text('{"where": "archived"}')
    assert resolve_corpus_path(both) == both
    assert "live" in resolve_corpus_path(both).read_text()


def test_a_file_absent_from_both_places_explains_both(corpus):
    missing = corpus / "data" / "nowhere.json"
    with pytest.raises(FileNotFoundError) as caught:
        resolve_corpus_path(missing)
    message = str(caught.value)
    assert "nowhere.json" in message and "archive" in message


def test_it_never_looks_outside_a_corpus(tmp_path):
    """A path with no research_artifacts component gets no archive treatment."""
    stray = tmp_path / "somewhere" / "else.json"
    assert corpus_path_or_none(stray) is None
    with pytest.raises(FileNotFoundError):
        resolve_corpus_path(stray)


def test_an_already_archived_path_is_not_archived_twice(corpus):
    doubly = corpus / "archive" / "data" / "absent.json"
    assert corpus_path_or_none(doubly) is None
    assert not (corpus / "archive" / "archive").exists()


def test_the_resolver_only_reads(corpus):
    """The receipts are evidence: resolving must not write, copy or normalize."""
    before = sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*"))
    resolve_corpus_path(corpus / "data" / "moved.json")
    corpus_path_or_none(corpus / "data" / "nowhere.json")
    after = sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*"))
    assert before == after


def test_the_guard_and_the_loaders_resolve_identically():
    """The original defect was a guard that answered differently from the code.

    conftest calls the same helper the loaders call. If that ever stops being
    true, tests get skipped or failed for a reason unrelated to the corpus.
    """
    import conftest

    source = __import__("inspect").getsource(conftest.research_corpus_available)
    assert "corpus_path_or_none" in source, (
        "the corpus guard has stopped resolving the way the loaders resolve"
    )

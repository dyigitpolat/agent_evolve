"""Shared fixtures: reset the harness registry and load in-tree integrations."""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

# Make the test-helper modules importable as top-level modules without adding
# the package root itself to sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_evolve import bootstrap
from agent_evolve.harness.registry import harness_registry


@pytest.fixture(autouse=True)
def _fresh_registry():
    harness_registry.clear()
    bootstrap._loaded = False
    bootstrap.load_integrations()
    yield
    harness_registry.clear()
    bootstrap._loaded = False


# ---------------------------------------------------------------------------
# The research corpus is not part of the distribution.
#
# Many tests here drive the paper's campaign scripts, which read fixtures from
# the research artifacts tree. Those files travel with the research, not with
# the package. Their absence used to raise FileNotFoundError *during
# collection*, which aborted the whole run: someone who cloned this repository
# alone could not run any test, including the ones needing nothing at all.
#
# Now a missing corpus removes those modules from collection and turns a
# corpus-missing failure inside a test into a skip. The suite a user can
# actually run stays runnable, and nothing silently passes: skips are reported.
# ---------------------------------------------------------------------------

RESEARCH_CORPUS = pathlib.Path(
    os.environ.get(
        "AGENTEVOLVE_RESEARCH_CORPUS",
        str(
            pathlib.Path(__file__).resolve().parents[2]
            / "papers"
            / "agent_evolve_aaai_2027"
            / "research_artifacts"
        ),
    )
)

#: Text that means a test module drives the research stack rather than the
#: shipped package. Detection is textual on purpose: several of these load a
#: campaign script through ``spec_from_file_location`` with a computed path, so
#: an import-graph rule misses them and a hand-kept list goes stale silently.
_RESEARCH_SIGNALS = (
    "research_artifacts",
    "examples/development",
    "examples/benchmarks",
    "examples.development",
    "examples.benchmarks",
    '"development"',
    "'development'",
    '"benchmarks"',
    "'benchmarks'",
)


#: Files the corpus-only modules read at *import* time. Presence of the corpus
#: directory is not enough, and assuming it was the defect: on 2026-07-28 the
#: research artifacts were split, moving these unmodified into ``archive/``
#: while the code kept pointing at the pre-split paths. The directory still
#: existed, so this guard switched itself off and five modules went back to
#: failing collection outright. A guard that checks the container rather than
#: the contents is the same species of check as the assertions this project
#: spent a week removing: it looks like a guarantee and cannot fail.
_REQUIRED_CORPUS_FILES = ("data/boils_v2_patch_native_legal_children.json",)


def research_corpus_available() -> bool:
    """True only when the corpus holds what the code under test actually reads.

    Resolves exactly the way the loaders resolve -- live path first, then the
    2026-07-28 archive location -- by calling the same helper they call. That
    shared call is the point. This guard's original defect was checking
    something *near* what the code reads (that the directory existed) instead
    of what it reads, and a guard that resolves paths its own way would be the
    same mistake wearing a different hat: it would drift from the loaders
    silently, and the only symptom would be tests skipped or failed for the
    wrong reason.
    """
    if not RESEARCH_CORPUS.is_dir():
        return False
    from examples.development.corpus_paths import corpus_path_or_none

    return all(
        corpus_path_or_none(RESEARCH_CORPUS / rel) is not None
        for rel in _REQUIRED_CORPUS_FILES
    )


def corpus_drift_reason():
    """Explain a present-but-unusable corpus, or return None."""
    if not RESEARCH_CORPUS.is_dir() or research_corpus_available():
        return None
    archived = [
        rel
        for rel in _REQUIRED_CORPUS_FILES
        if not (RESEARCH_CORPUS / rel).is_file()
        and (RESEARCH_CORPUS / "archive" / rel).is_file()
    ]
    if archived:
        return (
            "the research corpus is present but has drifted: "
            + ", ".join(archived)
            + " now live under archive/ after the 2026-07-28 split, while the "
            "campaign scripts still compute the pre-split path"
        )
    return "the research corpus is present but missing files the scripts read"


def pytest_ignore_collect(collection_path, config):
    """Without the corpus, do not even import a module that drives it.

    This used to be a hand-written list of five module names -- the five that
    failed in the environment it was written in, where the corpus was partly
    present. In a genuinely corpus-free environment eleven fail, so CI was
    always going to break on the six nobody had seen. Worse, some raise
    RuntimeError rather than FileNotFoundError, and all of them fail during
    *collection*, where the skip hook below cannot reach.

    Deriving the set from the same research signal used to mark the tests
    removes both problems: nothing to keep in sync, and a module added
    tomorrow is covered the day it appears.
    """
    if research_corpus_available():
        return None
    path = pathlib.Path(str(collection_path))
    if path.suffix == ".py" and path.name.startswith("test_") and _is_research_module(path):
        return True
    return None


#: A module that talks *about* the research corpus without needing one says so
#: with this line. Textual detection is deliberately broad, so it needs an
#: explicit escape rather than a cleverer rule -- a test for the corpus
#: resolver mentions every signal there is and requires none of them.
_CORPUS_FREE_PRAGMA = "corpus-free-by-construction"


def _is_research_module(path: pathlib.Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if _CORPUS_FREE_PRAGMA in text:
        return False
    return any(signal in text for signal in _RESEARCH_SIGNALS)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "research: drives the research stack; needs the corpus. Run with -m research.",
    )


def pytest_collection_modifyitems(config, items):
    """Split the suite: the shipped package by default, the research stack opt-in.

    The default run is what someone who cloned this repository can actually
    use, and it finishes in seconds. The research suite is a superset that
    needs the corpus and takes far longer; it is selected explicitly with
    ``-m research`` rather than being the thing everyone waits for.
    """
    marked = 0
    for item in items:
        path = pathlib.Path(str(getattr(item, "fspath", "") or ""))
        if path and _is_research_module(path):
            item.add_marker(pytest.mark.research)
            marked += 1
    if marked:
        config.stash.setdefault("agent_evolve_research_count", marked)


def _is_missing_corpus(exc: BaseException) -> bool:
    if not isinstance(exc, (FileNotFoundError, OSError)):
        return False
    text = str(getattr(exc, "filename", "") or "") + " " + str(exc)
    return "research_artifacts" in text or str(RESEARCH_CORPUS) in text


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """A test that fails only because the corpus is absent is skipped, not failed."""
    outcome = yield
    if research_corpus_available():
        return
    excinfo = outcome.excinfo
    if excinfo is not None and _is_missing_corpus(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception(
                f"needs the research corpus at {RESEARCH_CORPUS}, which is not "
                "part of the distribution",
                _use_item_location=True,
            )
        )

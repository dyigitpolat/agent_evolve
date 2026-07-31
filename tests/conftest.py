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

#: Modules that cannot even be imported without the corpus.
_CORPUS_ONLY_MODULES = (
    "test_run_boils_action_shadow_offline.py",
    "test_run_boils_agentic_pilot_v2_offline.py",
    "test_run_boils_local_oracle_offline.py",
    "test_run_boils_recombination_engine_v4_offline.py",
    "test_run_boils_recombination_v3_offline.py",
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

    Deliberately checks the *live* paths, not the archive. The scripts compute
    pre-split paths, so a file that exists only under ``archive/`` is one they
    cannot open: reporting it as available would restore exactly the failure
    this guard exists to convert into a skip. When this returns False and the
    directory is present, the corpus has drifted rather than gone missing --
    see ``corpus_drift_reason()``.
    """
    if not RESEARCH_CORPUS.is_dir():
        return False
    return all((RESEARCH_CORPUS / rel).is_file() for rel in _REQUIRED_CORPUS_FILES)


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


collect_ignore = [] if research_corpus_available() else list(_CORPUS_ONLY_MODULES)


def _is_research_module(path: pathlib.Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
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

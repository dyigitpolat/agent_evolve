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


def research_corpus_available() -> bool:
    return RESEARCH_CORPUS.is_dir()


collect_ignore = [] if research_corpus_available() else list(_CORPUS_ONLY_MODULES)


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

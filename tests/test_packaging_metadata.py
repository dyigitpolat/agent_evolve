"""The distribution metadata must say what the code actually supports.

Two classes of defect are pinned here, both of which had already shipped once:

1. **Version skew.** `pyproject.toml` carried a literal version and
   `agent_evolve.__init__` carried another. They disagreed for a release cycle,
   so `agent_evolve version` -- which reads `importlib.metadata` -- printed a
   number no source file contained. The fix is single-sourcing (`dynamic =
   ["version"]`, `attr = agent_evolve.__version__`); this test is what stops the
   literal from coming back.

2. **A declared range the code refuses to honour.** The Pydantic-AI boundary
   codec fails closed unless `pydantic_ai`, `pydantic` and `pydantic_core` are
   at the exact versions it recorded its replay contract against. The `llm`
   extra used to declare `pydantic-ai>=1.0,<2`, so a clean install resolved a
   newer patch release and 27 boundary tests failed on first run. The extra now
   pins the three constants, and this test asserts the two stay equal -- widen
   one without the other and the suite fails instead of the user's install.

These read `pyproject.toml` from the source tree, so they are skipped when the
package is exercised from an installed wheel with no project file beside it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import agent_evolve

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

pytestmark = pytest.mark.skipif(
    not PYPROJECT.is_file(),
    reason="no pyproject.toml beside the package; nothing to compare against",
)


@pytest.fixture(scope="module")
def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _pins(specifiers: list[str]) -> dict[str, str]:
    """{name: version} for every `name==version` specifier in the list."""
    found: dict[str, str] = {}
    for spec in specifiers:
        match = re.fullmatch(r"\s*([A-Za-z0-9_.-]+)\s*==\s*([0-9A-Za-z.+-]+)\s*", spec)
        if match:
            found[match.group(1).replace("_", "-").lower()] = match.group(2)
    return found


def test_version_is_single_sourced_from_the_code(project):
    metadata = project["project"]
    assert "version" not in metadata, (
        "pyproject.toml declares a literal version again. The version lives in "
        "agent_evolve.__version__ and the metadata must read it, or the two "
        "drift and `agent_evolve version` lies."
    )
    assert "version" in metadata.get("dynamic", []), (
        "the version must be declared dynamic so setuptools reads it from the code"
    )
    attr = project["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "agent_evolve.__version__", attr


def test_installed_distribution_version_matches_the_attribute():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("agent_evolve")
    except PackageNotFoundError:
        pytest.skip("agent_evolve is not installed as a distribution here")
    assert installed == agent_evolve.__version__, (
        f"distribution metadata says {installed} but agent_evolve.__version__ "
        f"is {agent_evolve.__version__}; reinstall, or stop writing the version "
        "in two places"
    )


def test_the_llm_extra_pins_exactly_what_the_boundary_codec_supports(project):
    codec = pytest.importorskip(
        "agent_evolve.integrations.pydantic_ai.boundary_codec",
        reason="the boundary codec needs pydantic-ai installed",
    )
    expected = {
        "pydantic-ai": codec.SUPPORTED_PYDANTIC_AI_VERSION,
        "pydantic": codec.SUPPORTED_PYDANTIC_VERSION,
        "pydantic-core": codec.SUPPORTED_PYDANTIC_CORE_VERSION,
    }
    extras = project["project"]["optional-dependencies"]
    for extra in ("llm", "pydantic_ai", "all"):
        declared = _pins(extras[extra])
        for name, wanted in expected.items():
            assert declared.get(name) == wanted, (
                f"extra '{extra}' declares {name}=={declared.get(name)!r} but "
                f"boundary_codec fails closed on anything except {wanted!r}. "
                "Either pin the metadata to the code, or widen the codec's "
                "replay contract -- which is a decision about what a sealed run "
                "means, not a packaging tweak."
            )


def test_the_installed_boundary_dependencies_are_the_supported_ones():
    """The environment this suite runs in must satisfy the pins it declares."""
    codec = pytest.importorskip(
        "agent_evolve.integrations.pydantic_ai.boundary_codec",
        reason="the boundary codec needs pydantic-ai installed",
    )
    import pydantic
    import pydantic_core
    import pydantic_ai

    assert (
        pydantic_ai.__version__,
        pydantic.__version__,
        pydantic_core.__version__,
    ) == (
        codec.SUPPORTED_PYDANTIC_AI_VERSION,
        codec.SUPPORTED_PYDANTIC_VERSION,
        codec.SUPPORTED_PYDANTIC_CORE_VERSION,
    ), (
        "this environment does not match the boundary codec's replay contract, "
        "so every boundary test will fail closed for a reason that has nothing "
        "to do with the code under test. Install the pinned extra: "
        "pip install 'agent_evolve[llm]'"
    )


def test_the_pymoo_extra_exists_because_the_swap_example_needs_it(project):
    extras = project["project"]["optional-dependencies"]
    assert "pymoo" in extras, (
        "examples/pymoo_swap/ and tests/test_pymoo_swap_acceptance.py need pymoo; "
        "an undeclared dev-venv dependency is how the swap demo rots"
    )
    assert any("pymoo" in spec for spec in extras["all"])


def test_the_console_entry_point_names_a_real_callable(project):
    scripts = project["project"]["scripts"]
    assert scripts == {"agent_evolve": "agent_evolve.cli:main"}, scripts
    from agent_evolve.cli import main

    assert callable(main)


def test_the_readme_named_in_the_metadata_exists_and_leads_with_the_guarantee(project):
    readme = PYPROJECT.parent / project["project"]["readme"]
    assert readme.is_file(), readme
    # The fallback guarantee is the product's thesis and the first thing a
    # reader must meet. Not a style rule: a README that leads with the model
    # oversells exactly what the measurements scope.
    # Whitespace-normalized so the assertion survives reflowing a paragraph.
    head = " ".join(readme.read_text(encoding="utf-8")[:2000].lower().split())
    assert "cannot do worse than the classical optimizer it replaces" in head, (
        "the README's opening no longer states the fallback guarantee: that "
        "the floor is the classical path. Everything else the README says is "
        "scoped to a venue; this is the one sentence that is not, and it is "
        "the product's thesis."
    )
    assert "nsga-ii" in head, "the README no longer names what it replaces"

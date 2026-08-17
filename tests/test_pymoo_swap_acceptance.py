"""The drop-in acceptance test: the pymoo -> agent_evolve swap, measured not asserted.

The package documents itself as a drop-in replacement for a classical MOEA. That
claim is only as true as the diff a user has to write, so this test *is* the
claim:

* :func:`test_the_swap_is_a_short_diff` recomputes the unified diff between
  ``examples/pymoo_swap/nsga2_baseline.py`` and its swapped twin
  ``agentevolve_swap.py`` and fails if it grows past the declared budget. It
  needs no pymoo, no credential and no network, so it runs in the default suite
  and cannot be skipped into irrelevance.
* :func:`test_both_arms_run_and_are_comparable` executes both scripts and checks
  that they report the same evaluation budget and a nonempty front in the same
  units. This one needs pymoo and is skipped without it.

Why a budget and not a golden diff: pinning the exact diff text would make every
harmless rename a failure, while pinning nothing would let the swap decay into a
walkthrough. The budget is the thing being claimed.

If a change to the public API makes the budget unreachable, raising the numbers
here is the wrong repair. The finding is about the API's genericity and belongs
in the API.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SWAP_DIR = REPO_ROOT / "examples" / "pymoo_swap"
BASELINE = SWAP_DIR / "nsga2_baseline.py"
SWAPPED = SWAP_DIR / "agentevolve_swap.py"

# The declared budget, in changed lines, one side of the diff.
#
#   1  the module docstring, which names the arm and therefore must differ
#   2  the optimizer imports -- you are importing a different optimizer
#   3  API contact: call it, read the evaluation count, read the front
#
# API_CONTACT_LINES is the number that carries the drop-in claim. The other
# three lines are not API surface, and are counted separately so that a change
# to a docstring can never be mistaken for a change to the swap's cost.
MAX_CHANGED_LINES = 6
API_CONTACT_LINES = 3


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _changed(baseline: list[str], swapped: list[str]) -> tuple[list[str], list[str]]:
    """Return (removed, added) content lines of the unified diff."""
    removed: list[str] = []
    added: list[str] = []
    for line in difflib.unified_diff(baseline, swapped, lineterm="", n=0):
        if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("-"):
            removed.append(line[1:])
        elif line.startswith("+"):
            added.append(line[1:])
    return removed, added


def _is_import(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("import ") or stripped.startswith("from ")


def _is_docstring(line: str) -> bool:
    return line.strip().startswith('"""')


def test_the_swap_files_exist_and_the_problem_is_shared():
    """The swap must not be two different problems wearing the same name."""
    assert BASELINE.is_file(), f"missing {BASELINE}"
    assert SWAPPED.is_file(), f"missing {SWAPPED}"
    for path in (BASELINE, SWAPPED):
        text = path.read_text(encoding="utf-8")
        assert 'get_problem("zdt1", n_var=4)' in text, (
            f"{path.name} no longer builds the shared pymoo problem; the two "
            "arms must optimize the same object for the diff to mean anything."
        )
        assert "from pymoo.problems import get_problem" in text, (
            f"{path.name} must obtain the problem from pymoo, unchanged: the "
            "claim is that the problem definition survives the swap."
        )


def test_the_swap_is_a_short_diff():
    baseline, swapped = _lines(BASELINE), _lines(SWAPPED)
    removed, added = _changed(baseline, swapped)

    rendered = "\n".join(
        difflib.unified_diff(
            baseline, swapped,
            fromfile=f"examples/pymoo_swap/{BASELINE.name}",
            tofile=f"examples/pymoo_swap/{SWAPPED.name}",
            lineterm="", n=1,
        )
    )
    detail = (
        f"\n\nthe measured swap diff ({len(removed)} removed / {len(added)} "
        f"added, budget {MAX_CHANGED_LINES} each way):\n{rendered}\n\n"
        "Raising MAX_CHANGED_LINES is not the repair. A swap that costs more "
        "lines is a finding about the public API's genericity."
    )

    assert removed and added, "the two arms are identical; there is no swap" + detail
    assert len(removed) <= MAX_CHANGED_LINES, detail
    assert len(added) <= MAX_CHANGED_LINES, detail

    for side, name in ((removed, "removed"), (added, "added")):
        contact = [
            line for line in side
            if line.strip() and not _is_import(line) and not _is_docstring(line)
        ]
        assert len(contact) == API_CONTACT_LINES, (
            f"{len(contact)} lines of API contact on the {name} side, expected "
            f"{API_CONTACT_LINES}: {contact}" + detail
        )


def test_the_swapped_arm_touches_only_the_documented_public_surface():
    """The swap may not reach past the public API into internals."""
    _, added = _changed(_lines(BASELINE), _lines(SWAPPED))
    imports = [line.strip() for line in added if _is_import(line)]
    assert imports == [
        "from agent_evolve import optimize",
        "from agent_evolve.integrations.pymoo_adapter import from_pymoo",
    ], (
        "the swap must import from the top-level package and the named "
        f"integration module only, got {imports}"
    )
    body = " ".join(line for line in added if not _is_import(line))
    assert "._" not in body and " _" not in body, (
        f"the swap reads a private attribute, which is not a drop-in: {body!r}"
    )


def _run(script: Path) -> dict[str, str]:
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"{script.name} exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    report: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            report[key.strip()] = value.strip()
    return report


def test_both_arms_run_and_are_comparable():
    pytest.importorskip(
        "pymoo",
        reason="the executable half of the swap test needs pymoo: "
               "pip install 'agent_evolve[pymoo]'",
    )
    classical = _run(BASELINE)
    swapped = _run(SWAPPED)

    assert classical["arm"] == BASELINE.stem
    assert swapped["arm"] == SWAPPED.stem

    # Comparable means the same budget spent and the same reported quantities,
    # not the same winner. Which arm wins is a measurement, not an acceptance
    # criterion, and this test deliberately does not assert one.
    assert classical["evaluations"] == swapped["evaluations"] == "60", (
        f"the arms did not spend the same budget: classical "
        f"{classical.get('evaluations')} vs swapped {swapped.get('evaluations')}"
    )
    assert set(classical) == set(swapped) == {
        "arm", "evaluations", "front size", "best f0", "best f1",
    }
    for arm, report in (("classical", classical), ("swapped", swapped)):
        assert int(report["front size"]) >= 1, f"{arm} returned an empty front"
        for key in ("best f0", "best f1"):
            value = float(report[key])
            assert value == value and abs(value) != float("inf"), (
                f"{arm} reported a non-finite {key}: {report[key]}"
            )

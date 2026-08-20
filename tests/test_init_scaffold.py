"""``agent_evolve init`` must write a file that is already a problem.

A scaffold is only worth shipping if what it writes runs. This one lands as a
valid ``Problem`` -- ``as_problem`` accepts it, ``diagnose`` can be pointed at
it the same minute -- with exactly one obligation left refusing by name, since
measuring is the thing nothing can guess for you. So these tests import the
written file rather than reading it, and then fill in that one blank and run a
real search through it.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from agent_evolve import as_problem, optimize
from agent_evolve.cli import main


def _write_scaffold(tmp_path, *argv) -> "object":
    assert main(["init", *argv]) == 0
    written = tmp_path / "problem_def.py"
    assert written.exists()
    return written


def _import_file(path):
    """Import the written file as its own module, off the filesystem."""
    spec = importlib.util.spec_from_file_location(f"scaffold_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_init_writes_problem_def_into_a_directory(tmp_path):
    written = _write_scaffold(tmp_path, str(tmp_path))
    text = written.read_text(encoding="utf-8")
    for obligation in ("candidate_model", "objectives", "seeds",
                       "validate", "materialize", "evaluate"):
        assert obligation in text, f"the template dropped {obligation}"
    assert "Describing your problem" in text, (
        "the template must point at the README section that explains it"
    )


def test_the_written_file_imports_and_is_a_problem(tmp_path):
    written = _write_scaffold(tmp_path, str(tmp_path))
    module = _import_file(written)
    problem = module.problem
    assert as_problem(problem) is problem
    assert [(o.name, o.goal) for o in problem.objectives] == [
        ("throughput", "max"), ("cost", "min")]
    assert list(problem.seeds()) == [{"workers": 8}]
    assert problem.validate({"workers": 8}).ok


def test_the_one_unwritten_obligation_refuses_by_name(tmp_path):
    # A template that returned a plausible number instead would let a run look
    # like it worked while measuring nothing.
    module = _import_file(_write_scaffold(tmp_path, str(tmp_path)))
    with pytest.raises(NotImplementedError) as raised:
        module.problem.evaluate((8,))
    assert "evaluate()" in str(raised.value)


def test_filling_the_blank_makes_it_a_working_search(tmp_path):
    module = _import_file(_write_scaffold(tmp_path, str(tmp_path)))

    def evaluate(self, artifact):
        (workers,) = artifact
        return {"throughput": float(workers), "cost": float(workers) ** 0.5}

    module.MyProblem.evaluate = evaluate
    result = optimize(module.MyProblem(), budget=8, proposer="random", seed=0)
    assert result.evaluations <= 8
    assert result.pareto_front
    assert set(result.best.objectives) == {"throughput", "cost"}


def test_init_refuses_to_overwrite(tmp_path):
    _write_scaffold(tmp_path, str(tmp_path))
    with pytest.raises(SystemExit) as raised:
        main(["init", str(tmp_path)])
    assert "refusing to overwrite" in str(raised.value)


def test_init_writes_the_py_path_it_is_given(tmp_path):
    target = tmp_path / "nested" / "my_problem.py"
    assert main(["init", str(target)]) == 0
    assert target.exists()
    assert _import_file(target).problem is not None


def test_init_defaults_to_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / "problem_def.py").exists()

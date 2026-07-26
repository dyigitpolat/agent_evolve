"""Provider- and PDE-free conformance for constructive Heat2D."""

from __future__ import annotations

import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import pytest
from pydantic import ValidationError

from agent_evolve.agentic import (
    AgenticBenchmark,
    FiniteVariationCatalog,
    FrozenJsonObject,
    thaw_json,
)
from examples.benchmarks.heat2d_constructive.agentic_benchmark import (
    benchmark,
    finite_variation_catalog,
)
from examples.benchmarks.heat2d_constructive.artifact_boundary import (
    decode_mapping,
    decoder_source_sha256,
    direct_v3_runner_sha256,
)
from examples.benchmarks.heat2d_constructive.candidate import (
    CandidateConfig,
    SEED_LAYOUT_A,
    SEED_LAYOUT_B,
    normalize_candidate,
    seed_layouts,
)
from examples.benchmarks.heat2d_constructive.finite_variation_catalog import (
    CATALOG_DEFINITION_SHA256,
    CATALOG_ID,
    LOCUS_GRIDS,
)
from examples.benchmarks.heat2d_constructive.problem_def import (
    DirectV3ContractError,
    DirectV3Evaluator,
    Heat2DConstructiveProblem,
    Heat2DDirectV3Evaluation,
    Heat2DDirectV3Settings,
    OBJECTIVE_NAME,
)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            path = key if not prefix else f"{prefix}.{key}"
            result.update(_flatten(item, path))
        return result
    return {prefix: value}


def _array_sha256(array: object) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _python_probe_stdout() -> str:
    return json.dumps(
        {
            "numpy_module_path": "/stable/environment/numpy/__init__.py",
            "numpy_module_sha256": "a" * 64,
            "numpy_version": "2.3.5",
            "python_base_prefix": "/stable/base",
            "python_prefix": "/stable/environment",
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def test_public_benchmark_is_scalar_objective_only_and_serialized() -> None:
    assert isinstance(benchmark, AgenticBenchmark)
    assert benchmark.detailed_evaluator is None
    assert benchmark.outcome_relation is None
    assert tuple((item.name, item.goal) for item in benchmark.objectives) == (
        (OBJECTIVE_NAME, "min"),
    )
    assert isinstance(finite_variation_catalog, FiniteVariationCatalog)
    assert benchmark.finite_variation_catalog_identities == (
        (CATALOG_ID, 1, CATALOG_DEFINITION_SHA256),
    )
    assert benchmark.problem.settings.external_concurrency == 1
    assert len(decoder_source_sha256()) == 64
    assert len(direct_v3_runner_sha256()) == 64


def test_fixed_candidate_schema_rejects_topology_and_numeric_drift() -> None:
    first, second = seed_layouts()
    assert type(first) is CandidateConfig
    assert type(second) is CandidateConfig

    for candidate in (first, second):
        mapping = candidate.decoder_mapping()
        assert [item["primitive_id"] for item in mapping["primitives"]] == [
            "trunk",
            "left_lobe",
            "right_bar",
            "central_hole",
        ]
        assert [item["kind"] for item in mapping["primitives"]] == [
            "capsule",
            "ellipse",
            "box",
            "ellipse",
        ]
        assert [item["operation"] for item in mapping["primitives"]] == [
            "add",
            "add",
            "add",
            "remove",
        ]

    extra = dict(SEED_LAYOUT_A)
    extra["primitives"] = []
    with pytest.raises(ValidationError, match="Extra inputs"):
        normalize_candidate(extra)
    boolean = json.loads(json.dumps(SEED_LAYOUT_A))
    boolean["material_fraction"] = True
    with pytest.raises(ValidationError):
        normalize_candidate(boolean)
    short = json.loads(json.dumps(SEED_LAYOUT_A))
    short["trunk"]["end_x"] = short["trunk"]["start_x"]
    short["trunk"]["end_y"] = short["trunk"]["start_y"]
    with pytest.raises(ValidationError, match="at least 0.02"):
        normalize_candidate(short)


def test_seed_phenotypes_are_deterministic_distinct_and_semantically_bound() -> None:
    pytest.importorskip("numpy")
    first, second = seed_layouts()
    decoded_first = decode_mapping(first.decoder_mapping(), resolution=1001)
    decoded_replay = decode_mapping(first.decoder_mapping(), resolution=1001)
    decoded_second = decode_mapping(second.decoder_mapping(), resolution=1001)
    assert decoded_first.raw_array_sha256 == decoded_replay.raw_array_sha256
    assert decoded_first.phenotype_sha256 == decoded_replay.phenotype_sha256
    assert decoded_first.raw_array_sha256 != decoded_second.raw_array_sha256
    assert decoded_first.phenotype_sha256 != decoded_second.phenotype_sha256
    assert benchmark.phenotype_identity.identify(
        first.model_dump(mode="python")
    ).value_sha256 == decoded_first.phenotype_sha256
    assert benchmark.phenotype_identity.identify(
        second.model_dump(mode="python")
    ).value_sha256 == decoded_second.phenotype_sha256


def test_finite_catalog_emits_at_least_eight_exact_safe_moves_per_locus() -> None:
    parent = normalize_candidate(SEED_LAYOUT_A)
    parent_payload = parent.model_dump(mode="python")
    contract = benchmark.bind_finite_variation(CATALOG_ID, parent_payload)
    counts = Counter(dict(option.metadata)["locus"] for option in contract.options)
    assert set(counts) == {locus for locus, _, _ in LOCUS_GRIDS}
    assert min(counts.values()) >= 8
    assert len(contract.options) == len(
        {option.child_configuration_sha256 for option in contract.options}
    )
    parent_flat = _flatten(parent_payload)
    for option in contract.options:
        assert type(option.child_configuration) is FrozenJsonObject
        child_payload = thaw_json(option.child_configuration)
        normalize_candidate(child_payload)
        differences = {
            key
            for key, value in _flatten(child_payload).items()
            if value != parent_flat[key]
        }
        assert differences == {dict(option.metadata)["locus"]}

    replay = benchmark.bind_finite_variation(CATALOG_ID, parent_payload)
    assert replay.identity_sha256 == contract.identity_sha256
    changed = normalize_candidate(SEED_LAYOUT_B)
    rebound = benchmark.bind_finite_variation(
        CATALOG_ID,
        changed.model_dump(mode="python"),
    )
    assert rebound.parent_configuration_sha256 != contract.parent_configuration_sha256
    assert rebound.identity_sha256 != contract.identity_sha256


class _FakeEvaluator:
    def evaluate(self, config: object) -> Heat2DDirectV3Evaluation:
        candidate = normalize_candidate(config)
        digest = hashlib.sha256(
            json.dumps(
                candidate.model_dump(mode="python"), sort_keys=True
            ).encode("ascii")
        ).hexdigest()
        return Heat2DDirectV3Evaluation(
            objective_values={OBJECTIVE_NAME: candidate.material_fraction},
            output_dir=Path("/tmp/provider-free"),
            genotype_sha256=digest,
            phenotype_sha256=digest,
            raw_array_sha256=digest,
            representation_spec_sha256=digest,
            finite_element_volume=candidate.material_fraction,
            grayness=0.1,
            gray_fraction_005_095=0.2,
            adapter_elapsed_s=0.0,
            evaluator_elapsed_s=0.0,
            elapsed_inside_container_s=0.0,
            queue_wait_s=0.0,
            peak_rss_bytes=1,
            manifest={},
        )


def test_problem_supports_provider_free_evaluator_injection(tmp_path: Path) -> None:
    settings = Heat2DDirectV3Settings(output_root=tmp_path)
    problem = Heat2DConstructiveProblem(settings, evaluator=_FakeEvaluator())
    assert problem.validate(SEED_LAYOUT_A) is True
    assert problem.evaluate(SEED_LAYOUT_A) == {OBJECTIVE_NAME: 0.45}
    assert problem.candidate_key(SEED_LAYOUT_A) != problem.candidate_key(SEED_LAYOUT_B)


def test_direct_v3_numpy_requirement_is_exactly_pinned(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned to exactly 2.3.5"):
        Heat2DDirectV3Settings(
            output_root=tmp_path,
            required_numpy_version="2.3.6",
        )


def test_direct_v3_preserves_symlinked_environment_interpreter_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    environment_bin = tmp_path / "ephemeral-environment" / "bin"
    environment_bin.mkdir(parents=True)
    invoked_python = environment_bin / "python"
    invoked_python.symlink_to(Path(__import__("sys").executable))
    probe_commands: list[tuple[str, ...]] = []
    help_commands: list[tuple[str, ...]] = []
    scientific_commands: list[tuple[str, ...]] = []

    class InvocationObserved(RuntimeError):
        pass

    def observe_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess:
        del kwargs
        if "-c" in command:
            probe_commands.append(command)
            return subprocess.CompletedProcess(
                command,
                0,
                _python_probe_stdout(),
                "",
            )
        if command[-1] == "--help":
            help_commands.append(command)
            return subprocess.CompletedProcess(
                command, 0, "usage: run_heat2d_direct_v3.py\n", ""
            )
        scientific_commands.append(command)
        raise InvocationObserved

    monkeypatch.setattr(
        "examples.benchmarks.heat2d_constructive.problem_def.shutil.which",
        lambda executable, **kwargs: f"/usr/bin/{executable}",
    )
    monkeypatch.setattr(
        "agent_evolve.infrastructure.subprocess_boundary.subprocess.run",
        observe_run,
    )
    evaluator = DirectV3Evaluator(
        Heat2DDirectV3Settings(
            output_root=tmp_path / "outputs",
            resolution=41,
            python_executable=invoked_python,
        )
    )

    preflight = evaluator.preflight()
    assert preflight["python_executable_resolved"] == str(
        invoked_python.resolve(strict=True)
    )
    assert preflight["process_boundary"]["policy"]["argv_policy"] == (
        "preserve_exact_sequence_no_shell_v1"
    )
    assert preflight["required_dependencies"]["numpy"]["observed_version"] == (
        "2.3.5"
    )
    assert probe_commands[0][0] == str(invoked_python)
    assert help_commands[0][0] == str(invoked_python)
    with pytest.raises(InvocationObserved):
        evaluator.evaluate(SEED_LAYOUT_A)
    assert len(probe_commands) == 1
    assert len(help_commands) == 1
    assert scientific_commands[0][0] == str(invoked_python)


def test_direct_v3_preflight_identity_is_stable_across_environment_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment_paths = tuple(
        tmp_path / name / "bin" / "python" for name in ("environment-a", "environment-b")
    )
    for path in environment_paths:
        path.parent.mkdir(parents=True)
        path.symlink_to(Path(__import__("sys").executable))
    observed_executables: list[str] = []

    def probe(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess:
        del kwargs
        observed_executables.append(command[0])
        if command[-1] == "--help":
            return subprocess.CompletedProcess(
                command, 0, "usage: run_heat2d_direct_v3.py\n", ""
            )
        return subprocess.CompletedProcess(
            command,
            0,
            _python_probe_stdout(),
            "",
        )

    monkeypatch.setattr(
        "examples.benchmarks.heat2d_constructive.problem_def.shutil.which",
        lambda executable, **kwargs: f"/usr/bin/{executable}",
    )
    monkeypatch.setattr(
        "agent_evolve.infrastructure.subprocess_boundary.subprocess.run",
        probe,
    )
    receipts = tuple(
        DirectV3Evaluator(
            Heat2DDirectV3Settings(
                output_root=tmp_path / f"outputs-{index}",
                python_executable=path,
            )
        ).preflight()
        for index, path in enumerate(environment_paths)
    )

    assert receipts[0] == receipts[1]
    assert observed_executables == [
        str(environment_paths[0]),
        str(environment_paths[0]),
        str(environment_paths[1]),
        str(environment_paths[1]),
    ]


def test_direct_v3_rejects_failed_dependency_probe_without_leaking_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invoked_python = tmp_path / "environment" / "bin" / "python"
    invoked_python.parent.mkdir(parents=True)
    invoked_python.symlink_to(Path(__import__("sys").executable))
    secret = "sensitive-child-stderr"

    monkeypatch.setattr(
        "examples.benchmarks.heat2d_constructive.problem_def.shutil.which",
        lambda executable, **kwargs: f"/usr/bin/{executable}",
    )
    monkeypatch.setattr(
        "agent_evolve.infrastructure.subprocess_boundary.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", secret
        ),
    )
    evaluator = DirectV3Evaluator(
        Heat2DDirectV3Settings(
            output_root=tmp_path / "outputs",
            python_executable=invoked_python,
        )
    )
    with pytest.raises(DirectV3ContractError, match="dependency probe failed") as info:
        evaluator.preflight()
    assert secret not in str(info.value)


def test_direct_v3_rejects_non_executable_python_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    non_executable = tmp_path / "python"
    non_executable.write_text("not an executable", encoding="utf-8")
    non_executable.chmod(0o600)
    monkeypatch.setattr(
        "examples.benchmarks.heat2d_constructive.problem_def.shutil.which",
        lambda executable, **kwargs: f"/usr/bin/{executable}",
    )
    evaluator = DirectV3Evaluator(
        Heat2DDirectV3Settings(
            output_root=tmp_path / "outputs",
            python_executable=non_executable,
        )
    )
    with pytest.raises(DirectV3ContractError, match="not executable"):
        evaluator.preflight()


def test_direct_v3_wrapper_uses_unique_dirs_and_serializes_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    active = 0
    maximum_active = 0
    guard = threading.Lock()

    def fake_run(command, **kwargs):
        nonlocal active, maximum_active
        del kwargs
        if "-c" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                _python_probe_stdout(),
                "",
            )
        if command[-1] == "--help":
            return subprocess.CompletedProcess(
                command, 0, "usage: run_heat2d_direct_v3.py\n", ""
            )
        with guard:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            time.sleep(0.03)
            design_path = Path(command[command.index("--design") + 1])
            output_dir = Path(command[command.index("--output-dir") + 1])
            resolution = int(command[command.index("--resolution") + 1])
            expected_volume = float(
                command[command.index("--expected-fe-volume") + 1]
            )
            design = np.load(design_path, allow_pickle=False)
            output_dir.mkdir(parents=True)
            manifest = {
                "schema_version": 3,
                "evaluator_id": "engibench-heatconduction2d-direct-v3",
                "mode": "forward",
                "elapsed_s": 0.02,
                "full_pde_solve_count": 1,
                "candidate": {
                    "resolution": resolution,
                    "raw_array_sha256": _array_sha256(design),
                    "expected_finite_element_volume": expected_volume,
                },
                "container_result": {
                    "result": {OBJECTIVE_NAME: expected_volume / 1000.0},
                    "elapsed_inside_container_s": 0.01,
                    "resource_measurement": {
                        "peak_rss_bytes_by_linux_kib_convention": 1024,
                    },
                },
                "all_checks_pass": True,
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            return subprocess.CompletedProcess(command, 0, "{}", "")
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(
        "agent_evolve.infrastructure.subprocess_boundary.subprocess.run",
        fake_run,
    )
    settings = Heat2DDirectV3Settings(
        output_root=tmp_path,
        resolution=41,
        python_executable=Path(__import__("sys").executable),
    )
    evaluator = DirectV3Evaluator(settings)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(evaluator.evaluate, (SEED_LAYOUT_A, SEED_LAYOUT_B))
        )
    assert maximum_active == 1
    assert results[0].output_dir != results[1].output_dir
    assert all(result.output_dir.is_dir() for result in results)
    assert {result.objective_values[OBJECTIVE_NAME] for result in results} == {
        0.00045,
        0.00038,
    }


def test_adapter_framework_imports_use_only_public_agentic_facade() -> None:
    adapter_dir = Path(__file__).parents[1] / "examples/benchmarks/heat2d_constructive"
    for path in adapter_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("agent_evolve")
        }
        assert modules <= {"agent_evolve.agentic"}, (path.name, modules)

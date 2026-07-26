"""Thin, fail-closed boundary to the qualified Heat2D research artifacts.

The numerical decoder and direct evaluator predate this benchmark package and
remain their own authorities under ``papers/.../research_artifacts/scripts``.
This module locates and loads the decoder without copying its implementation.
An installed deployment may set ``AGENT_EVOLVE_HEAT2D_ARTIFACT_SCRIPTS`` to
the exact artifact directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any


_SCRIPTS_ENV = "AGENT_EVOLVE_HEAT2D_ARTIFACT_SCRIPTS"
_DECODER_FILENAME = "heat2d_constructive_layout.py"
_INTEGRITY_FILENAME = "heat2d_integrity_adapter.py"
_RUNNER_FILENAME = "run_heat2d_direct_v3.py"
_DECODER_MODULE_NAME = "_agent_evolve_heat2d_constructive_layout"
_LOAD_LOCK = threading.Lock()
_DECODER_MODULE: ModuleType | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_scripts_dir() -> Path:
    """Resolve the exact local artifact directory or fail with instructions."""

    override = os.environ.get(_SCRIPTS_ENV)
    if override is not None:
        if not override.strip() or override != override.strip():
            raise RuntimeError(f"{_SCRIPTS_ENV} must be a canonical non-empty path")
        candidate = Path(override).expanduser().resolve()
    else:
        repository_root = Path(__file__).resolve().parents[4]
        candidate = (
            repository_root
            / "papers"
            / "agent_evolve_aaai_2027"
            / "research_artifacts"
            / "scripts"
        ).resolve()
    required = (_DECODER_FILENAME, _INTEGRITY_FILENAME, _RUNNER_FILENAME)
    missing = tuple(name for name in required if not (candidate / name).is_file())
    if missing:
        raise RuntimeError(
            "Heat2D developmental artifact directory is incomplete: "
            f"directory={candidate}, missing={missing}"
        )
    return candidate


def decoder_source_sha256() -> str:
    return _sha256_file(artifact_scripts_dir() / _DECODER_FILENAME)


def direct_v3_runner_path() -> Path:
    return artifact_scripts_dir() / _RUNNER_FILENAME


def direct_v3_runner_sha256() -> str:
    return _sha256_file(direct_v3_runner_path())


def _load_decoder() -> ModuleType:
    global _DECODER_MODULE
    with _LOAD_LOCK:
        if _DECODER_MODULE is not None:
            return _DECODER_MODULE
        scripts = artifact_scripts_dir()
        decoder_path = scripts / _DECODER_FILENAME

        # The decoder intentionally imports its sibling integrity adapter by
        # module name.  Refuse a foreign preloaded module rather than allowing
        # Python import order to choose numerical semantics.
        existing = sys.modules.get("heat2d_integrity_adapter")
        if existing is not None:
            loaded_path = Path(getattr(existing, "__file__", "")).resolve()
            expected_path = (scripts / _INTEGRITY_FILENAME).resolve()
            if loaded_path != expected_path:
                raise RuntimeError(
                    "foreign heat2d_integrity_adapter is already loaded: "
                    f"expected={expected_path}, observed={loaded_path}"
                )

        spec = importlib.util.spec_from_file_location(
            _DECODER_MODULE_NAME,
            decoder_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not construct the Heat2D decoder module spec")
        module = importlib.util.module_from_spec(spec)
        original_path = tuple(sys.path)
        sys.modules[_DECODER_MODULE_NAME] = module
        try:
            sys.path.insert(0, str(scripts))
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_DECODER_MODULE_NAME, None)
            raise
        finally:
            sys.path[:] = original_path
        if Path(getattr(module, "__file__", "")).resolve() != decoder_path.resolve():
            raise RuntimeError("Heat2D decoder was not loaded from the selected artifact")
        _DECODER_MODULE = module
        return module


def decode_mapping(candidate_mapping: dict[str, object], *, resolution: int) -> Any:
    """Decode through the qualified implementation and return its receipt."""

    module = _load_decoder()
    return module.decode_layout(candidate_mapping, resolution=resolution)


def representation_specification() -> dict[str, Any]:
    module = _load_decoder()
    specification = module.representation_specification()
    if not isinstance(specification, dict):
        raise RuntimeError("Heat2D decoder returned an invalid specification")
    return specification


def genotype_sha256(candidate_mapping: dict[str, object]) -> str:
    module = _load_decoder()
    layout = module.parse_layout(candidate_mapping)
    identity = module.genotype_sha256(layout)
    if not isinstance(identity, str) or len(identity) != 64:
        raise RuntimeError("Heat2D decoder returned an invalid genotype digest")
    return identity


def save_design(path: Path, decoded: Any) -> None:
    """Persist the decoder-owned float64 field without adding a second codec."""

    module = _load_decoder()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Heat2D design input {path}")
    module.np.save(path, decoded.simulator_design(), allow_pickle=False)


__all__ = [
    "artifact_scripts_dir",
    "decode_mapping",
    "decoder_source_sha256",
    "direct_v3_runner_path",
    "direct_v3_runner_sha256",
    "genotype_sha256",
    "representation_specification",
    "save_design",
]

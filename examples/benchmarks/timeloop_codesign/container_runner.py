"""Pinned in-container Timeloop evaluator.

This module intentionally uses only the Python standard library plus the
Timeloop installation already sealed inside the image.  It is invoked as a
process; importing it on the host performs no simulator work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


SCHEMA_VERSION = 1
EVALUATOR_ID = "timeloop-accelerator-codesign-2024-dse-v1"
SEARCH_SIZE = 2_000
MAPPER_THREADS = 1
MAPPER_ALGORITHM = "random_pruned"
OPTIMIZATION_METRICS = ("edp",)
_CANDIDATE_HASH_DOMAIN = b"agent-evolve:timeloop-codesign-candidate:v1\x00"

ASSET_ROOT = Path(
    "/usr/local/src/timeloop-python/tests/timeloop-accelergy-exercises/"
    "workspace/tutorial_exercises/02_interface_and_design_space_exploration_2024"
)
ASSET_SHA256 = {
    "/usr/local/bin/timeloop-mapper": (
        "863dea53d87dfc9f6965fbdf50103fdd5058829c84ab1906a76bd072f397f65c"
    ),
    "top.yaml.jinja": (
        "603d41cacb09e6de6542fd3531c1132feeb295b1cf7228a8c19e3b8a1fb25fe6"
    ),
    "inputs/arch.yaml": (
        "bea3a3fa0b11fddefc61d87d5a46918b8698604e53058e1bc4784171ae891bfc"
    ),
    "inputs/components.yaml": (
        "689dfaf38462dda6c50124fceac1cd9b9000bfa502e8a9e66bbb80ac98fac197"
    ),
    "inputs/mapper.yaml": (
        "12cede2db935aecfc31d1df61df670019a21dc66577642f75e9601f67462b34d"
    ),
    "inputs/problem.yaml": (
        "1fb86a8f8553f691ece11af43372a82955de19c4eb72e2239decc3599d41ac78"
    ),
    "inputs/variables.yaml": (
        "dbb2af5040105976bb456d2042d797786b45c4c77532fafab119aa4d4cfae611"
    ),
}

_ALLOWED = {
    "global_buffer_depth": (256, 512, 1024, 2048),
    "global_buffer_width": (64, 128, 256),
    "pe_mesh_x": (4, 8, 16, 32),
    "datawidth_bits": (8, 16),
}
_CANDIDATE_KEYS = frozenset((*_ALLOWED, "register_enabled"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
        object_pairs_hook=_strict_object,
    )


def _validate_candidate(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("candidate must be an exact JSON object")
    if set(value) != _CANDIDATE_KEYS:
        raise ValueError("candidate has missing or extra fields")
    result: dict[str, object] = {}
    for field, allowed in _ALLOWED.items():
        item = value[field]
        if type(item) is not int or item not in allowed:
            raise ValueError(f"invalid {field}")
        result[field] = item
    enabled = value["register_enabled"]
    if type(enabled) is not bool:
        raise ValueError("register_enabled must be an exact boolean")
    result["register_enabled"] = enabled
    return result


def _canonical_candidate_bytes(candidate: dict[str, object]) -> bytes:
    return json.dumps(
        candidate,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _candidate_sha256(candidate: dict[str, object]) -> str:
    return hashlib.sha256(
        _CANDIDATE_HASH_DOMAIN + _canonical_candidate_bytes(candidate)
    ).hexdigest()


def _verify_assets() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in ASSET_SHA256.items():
        path = Path(name) if name.startswith("/") else ASSET_ROOT / name
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned Timeloop asset drift: {name}")
        observed[name] = actual
    return observed


def _finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise RuntimeError(f"{name} is not finite and positive")
    return number


def evaluate(input_path: Path, output_dir: Path) -> dict[str, object]:
    candidate = _validate_candidate(_load_json(input_path))
    assets = _verify_assets()
    output_dir.mkdir(parents=True, exist_ok=False)

    # The image does not publish /usr/local/lib through its default loader
    # configuration when the notebook entrypoint is bypassed.
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
    from pytimeloop.timeloopfe import v4 as tl  # noqa: PLC0415

    os.chdir(ASSET_ROOT)
    specification = tl.Specification.from_yaml_files(
        "top.yaml.jinja",
        jinja_parse_data={
            "datawidth_jinja": candidate["datawidth_bits"],
            "reg_enabled": candidate["register_enabled"],
        },
    )
    specification.architecture.find("buffer").attributes["depth"] = candidate[
        "global_buffer_depth"
    ]
    specification.architecture.find("buffer").attributes["width"] = candidate[
        "global_buffer_width"
    ]
    specification.architecture.find("PE").spatial.meshX = candidate["pe_mesh_x"]
    specification.mapper.search_size = SEARCH_SIZE
    specification.mapper.num_threads = MAPPER_THREADS
    if specification.mapper.algorithm != MAPPER_ALGORITHM:
        raise RuntimeError("pinned mapper algorithm drift")
    if tuple(specification.mapper.optimization_metrics) != OPTIMIZATION_METRICS:
        raise RuntimeError("pinned mapper optimization metric drift")

    started = time.perf_counter()
    stats = tl.call_mapper(
        specification,
        output_dir=str(output_dir),
        log_to=str(output_dir / "output.log"),
    )
    elapsed_s = time.perf_counter() - started
    mapping_path = output_dir / "timeloop-mapper.map.yaml"
    if not mapping_path.is_file():
        raise RuntimeError("Timeloop did not publish its selected mapping")

    objectives = {
        "energy_joules": _finite_positive(stats.energy, "energy"),
        "latency_seconds": _finite_positive(stats.latency, "latency"),
        "area_square_meters": _finite_positive(stats.area, "area"),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "candidate": candidate,
        "candidate_sha256": _candidate_sha256(candidate),
        "objectives": objectives,
        "diagnostics": {
            "elapsed_s": _finite_positive(elapsed_s, "elapsed_s"),
            "cycles": int(stats.cycles),
            "computes": int(stats.computes),
            "mapping_sha256": _sha256(mapping_path),
        },
        "protocol": {
            "search_size": SEARCH_SIZE,
            "mapper_threads": MAPPER_THREADS,
            "mapper_algorithm": MAPPER_ALGORITHM,
            "optimization_metrics": list(OPTIMIZATION_METRICS),
        },
        "provenance": {
            "asset_sha256": assets,
            "runner_sha256": _sha256(Path(__file__)),
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--failure", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.input, args.output_dir)
        _atomic_json(args.result, result)
        return 0
    except Exception as error:  # process boundary must publish terminal evidence
        _atomic_json(
            args.failure,
            {
                "schema_version": SCHEMA_VERSION,
                "evaluator_id": EVALUATOR_ID,
                "error_type": type(error).__name__,
                "message": str(error)[:2_000],
                "traceback": traceback.format_exc()[-8_000:],
            },
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

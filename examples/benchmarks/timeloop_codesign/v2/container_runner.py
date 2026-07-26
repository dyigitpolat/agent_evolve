"""Pinned in-container evaluator for one three-medoid Timeloop v2 bundle.

Only the Python standard library and the Timeloop installation sealed in the
container image are used here.  The host compiles model-authored decisions to
closed runtime manifests before invoking this process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback


SCHEMA_VERSION = 1
EVALUATOR_ID = "timeloop-network-mapspace-policy-codesign-v2"
SEARCH_SIZE = 400
MAPPER_THREADS = 1
MAPPER_ALGORITHM = "random_pruned"
MAX_CONSECUTIVE_INVALID_MAPPINGS = 100_000
OPTIMIZATION_METRICS = ("edp",)
_MANIFEST_HASH_DOMAIN = b"agent-evolve:timeloop-v2-runtime-layer-manifest:v1\x00"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VALID_MAPPING_COUNT = re.compile(r"STATEMENT: ([0-9]+) valid mappings found")
_INVALID_MAPPING_LIMIT = re.compile(
    r"STATEMENT: ([0-9]+) invalid mappings .* terminating search"
)

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

_BUNDLE_KEYS = {
    "schema_version",
    "evaluator_id",
    "candidate_sha256",
    "compiled_plan_sha256",
    "panel_sha256",
    "layer_manifest_sha256",
    "layer_manifests",
    "protocol",
}
_MANIFEST_KEYS = {
    "schema_version",
    "translator_id",
    "translator_version",
    "translator_definition_sha256",
    "template_id",
    "template_sha256",
    "compiled_plan_sha256",
    "candidate_sha256",
    "panel_sha256",
    "medoid_ordinal",
    "layer_multiplicity",
    "architecture",
    "problem_instance",
    "constraints",
}
_ARCHITECTURE_KEYS = {
    "global_buffer_depth",
    "global_buffer_width",
    "pe_mesh_x",
    "datawidth_bits",
    "register_enabled",
}
_PROBLEM_KEYS = {
    "N",
    "C",
    "M",
    "R",
    "S",
    "P",
    "Q",
    "Hstride",
    "Wstride",
    "Hdilation",
    "Wdilation",
    "G",
    "H",
    "W",
    "Hpad",
    "Wpad",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _manifest_sha256(manifest: dict[str, object]) -> str:
    return hashlib.sha256(
        _MANIFEST_HASH_DOMAIN + _canonical_bytes(manifest)
    ).hexdigest()


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


def _exact_dict(value: object, keys: set[str], name: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{name} has missing or extra fields")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256_value(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is not numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{name} must be finite and positive")
    return number


def _verify_assets() -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, expected in ASSET_SHA256.items():
        path = Path(name) if name.startswith("/") else ASSET_ROOT / name
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"pinned Timeloop asset drift: {name}")
        observed[name] = actual
    return observed


def _validate_constraint(value: object, ordinal: int, register_enabled: bool) -> None:
    if type(value) is not dict:
        raise ValueError("runtime constraint must be an exact object")
    expected_type = ("temporal", "spatial", "temporal", "dataspace")[ordinal]
    if value.get("type") != expected_type:
        raise ValueError("runtime constraint order/type drift")
    if ordinal == 0:
        if set(value) != {"type", "target", "permutation", "no_reuse"}:
            raise ValueError("local temporal constraint shape drift")
        expected_target = "reg" if register_enabled else "PE"
        if value["target"] != expected_target:
            raise ValueError("local temporal target drift")
    elif ordinal == 1:
        if set(value) != {
            "type",
            "target",
            "permutation",
            "factors",
            "minimum_parallelism",
            "maximum_parallelism",
        }:
            raise ValueError("spatial constraint shape drift")
        if value["target"] != "PE":
            raise ValueError("spatial target drift")
    elif ordinal == 2:
        if set(value) != {"type", "target", "permutation", "no_reuse"}:
            raise ValueError("global temporal constraint shape drift")
        if value["target"] != "buffer":
            raise ValueError("global temporal target drift")
    else:
        if set(value) != {"type", "target", "keep", "bypass"}:
            raise ValueError("dataspace constraint shape drift")
        if value["target"] != "buffer":
            raise ValueError("dataspace target drift")


def _validate_manifest(
    value: object,
    ordinal: int,
    bundle: dict[str, object],
) -> dict[str, object]:
    manifest = _exact_dict(value, _MANIFEST_KEYS, f"layer_manifest[{ordinal}]")
    if manifest["schema_version"] != 1 or manifest["medoid_ordinal"] != ordinal:
        raise ValueError("runtime manifest schema or ordinal drift")
    if manifest["candidate_sha256"] != bundle["candidate_sha256"]:
        raise ValueError("runtime candidate provenance drift")
    if manifest["compiled_plan_sha256"] != bundle["compiled_plan_sha256"]:
        raise ValueError("runtime compiled-plan provenance drift")
    if manifest["panel_sha256"] != bundle["panel_sha256"]:
        raise ValueError("runtime panel provenance drift")
    _positive_int(manifest["layer_multiplicity"], "layer_multiplicity")
    architecture = _exact_dict(
        manifest["architecture"], _ARCHITECTURE_KEYS, "architecture"
    )
    for key in _ARCHITECTURE_KEYS - {"register_enabled"}:
        _positive_int(architecture[key], key)
    if type(architecture["register_enabled"]) is not bool:
        raise ValueError("register_enabled must be an exact boolean")
    problem = _exact_dict(manifest["problem_instance"], _PROBLEM_KEYS, "problem")
    for key, item in problem.items():
        if type(item) is not int or (item <= 0 and key not in {"Hpad", "Wpad"}):
            raise ValueError(f"invalid problem dimension: {key}")
    if problem["Hpad"] != 0 or problem["Wpad"] != 0:
        raise ValueError("only zero padding coefficients are supported")
    constraints = manifest["constraints"]
    if type(constraints) is not list or len(constraints) != 4:
        raise ValueError("runtime manifest requires four constraints")
    for constraint_ordinal, constraint in enumerate(constraints):
        _validate_constraint(
            constraint,
            constraint_ordinal,
            architecture["register_enabled"],
        )
    return manifest


def _validate_bundle(value: object) -> dict[str, object]:
    bundle = _exact_dict(value, _BUNDLE_KEYS, "bundle")
    if bundle["schema_version"] != SCHEMA_VERSION:
        raise ValueError("bundle schema drift")
    if bundle["evaluator_id"] != EVALUATOR_ID:
        raise ValueError("evaluator identity drift")
    for key in ("candidate_sha256", "compiled_plan_sha256", "panel_sha256"):
        _sha256_value(bundle[key], key)
    hashes = bundle["layer_manifest_sha256"]
    manifests = bundle["layer_manifests"]
    if type(hashes) is not list or len(hashes) != 3:
        raise ValueError("bundle requires three manifest digests")
    if type(manifests) is not list or len(manifests) != 3:
        raise ValueError("bundle requires three layer manifests")
    for ordinal, (digest, raw_manifest) in enumerate(zip(hashes, manifests)):
        _sha256_value(digest, f"layer_manifest_sha256[{ordinal}]")
        manifest = _validate_manifest(raw_manifest, ordinal, bundle)
        if _manifest_sha256(manifest) != digest:
            raise ValueError("runtime layer manifest digest mismatch")
    protocol = _exact_dict(
        bundle["protocol"],
        {
            "search_size",
            "mapper_threads",
            "mapper_algorithm",
            "max_consecutive_invalid_mappings",
            "optimization_metrics",
        },
        "protocol",
    )
    if (
        protocol["search_size"] != SEARCH_SIZE
        or protocol["mapper_threads"] != MAPPER_THREADS
        or protocol["mapper_algorithm"] != MAPPER_ALGORITHM
        or protocol["max_consecutive_invalid_mappings"]
        != MAX_CONSECUTIVE_INVALID_MAPPINGS
        or protocol["optimization_metrics"] != list(OPTIMIZATION_METRICS)
    ):
        raise ValueError("frozen mapper protocol drift")
    return bundle


def _replace_constraint(node: object, key: str, payload: dict[str, object]) -> None:
    constraints = node.constraints
    old = constraints[key]
    new = type(old)(payload)
    constraints[key] = new
    setattr(constraints, key, new)


def _apply_manifest(manifest: dict[str, object], tl: object) -> object:
    architecture = manifest["architecture"]
    constraints = manifest["constraints"]
    specification = tl.Specification.from_yaml_files(
        "top.yaml.jinja",
        jinja_parse_data={
            "datawidth_jinja": architecture["datawidth_bits"],
            "reg_enabled": architecture["register_enabled"],
        },
    )
    buffer = specification.architecture.find("buffer")
    pe = specification.architecture.find("PE")
    local = (
        specification.architecture.find("reg")
        if architecture["register_enabled"]
        else pe
    )
    buffer.attributes["depth"] = architecture["global_buffer_depth"]
    buffer.attributes["width"] = architecture["global_buffer_width"]
    pe.spatial.meshX = architecture["pe_mesh_x"]
    specification.problem.instance.update(manifest["problem_instance"])

    local_temporal, spatial, global_temporal, dataspace = constraints
    _replace_constraint(
        local,
        "temporal",
        {
            "permutation": local_temporal["permutation"],
            "no_reuse": local_temporal["no_reuse"],
        },
    )
    _replace_constraint(
        pe,
        "spatial",
        {
            "permutation": spatial["permutation"],
            "factors": spatial["factors"],
        },
    )
    _replace_constraint(
        buffer,
        "temporal",
        {
            "permutation": global_temporal["permutation"],
            "no_reuse": global_temporal["no_reuse"],
        },
    )
    _replace_constraint(
        buffer,
        "dataspace",
        {"keep": dataspace["keep"], "bypass": dataspace["bypass"]},
    )
    specification.mapper.search_size = SEARCH_SIZE
    specification.mapper.num_threads = MAPPER_THREADS
    specification.mapper.timeout = MAX_CONSECUTIVE_INVALID_MAPPINGS
    if specification.mapper.algorithm != MAPPER_ALGORITHM:
        raise RuntimeError("pinned mapper algorithm drift")
    if tuple(specification.mapper.optimization_metrics) != OPTIMIZATION_METRICS:
        raise RuntimeError("pinned mapper metric drift")

    # Typed front-end projection check: every host constraint must survive the
    # pinned parser/object boundary exactly before the native mapper sees it.
    if tuple(local.constraints.temporal.permutation) != tuple(
        local_temporal["permutation"]
    ) or tuple(local.constraints.temporal.no_reuse) != tuple(
        local_temporal["no_reuse"]
    ):
        raise RuntimeError("local temporal front-end projection drift")
    if tuple(pe.constraints.spatial.permutation) != tuple(
        spatial["permutation"]
    ) or tuple(pe.constraints.spatial.factors) != tuple(spatial["factors"]):
        raise RuntimeError("spatial front-end projection drift")
    if tuple(buffer.constraints.temporal.permutation) != tuple(
        global_temporal["permutation"]
    ):
        raise RuntimeError("global temporal front-end projection drift")
    if tuple(buffer.constraints.dataspace.keep) != tuple(dataspace["keep"]) or tuple(
        buffer.constraints.dataspace.bypass
    ) != tuple(dataspace["bypass"]):
        raise RuntimeError("dataspace front-end projection drift")
    return specification


def _evaluate_layer(
    manifest: dict[str, object],
    manifest_sha256: str,
    output_dir: Path,
    tl: object,
) -> dict[str, object]:
    specification = _apply_manifest(manifest, tl)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "output.log"
    started = time.perf_counter()
    stats = tl.call_mapper(
        specification,
        output_dir=str(output_dir),
        log_to=str(log_path),
    )
    elapsed_s = time.perf_counter() - started
    mapping_path = output_dir / "timeloop-mapper.map.yaml"
    processed_path = output_dir / "parsed-processed-input.yaml"
    if not mapping_path.is_file() or not processed_path.is_file():
        raise RuntimeError("Timeloop did not publish mapping/input evidence")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    completed_matches = _VALID_MAPPING_COUNT.findall(log_text)
    invalid_matches = _INVALID_MAPPING_LIMIT.findall(log_text)
    if len(completed_matches) + len(invalid_matches) != 1:
        raise RuntimeError("could not account for the mapper termination reason")
    completed = bool(completed_matches)
    reported_valid_count = int(completed_matches[0]) if completed else None
    consecutive_invalid_count = int(invalid_matches[0]) if invalid_matches else 0
    return {
        "medoid_ordinal": manifest["medoid_ordinal"],
        "layer_multiplicity": manifest["layer_multiplicity"],
        "layer_manifest_sha256": manifest_sha256,
        "energy_joules": _positive_float(stats.energy, "energy"),
        "latency_seconds": _positive_float(stats.latency, "latency"),
        "area_square_meters": _positive_float(stats.area, "area"),
        "cycles": _positive_int(int(stats.cycles), "cycles"),
        "computes": _positive_int(int(stats.computes), "computes"),
        "requested_valid_mapping_count": SEARCH_SIZE,
        "reported_valid_mapping_count": reported_valid_count,
        "consecutive_invalid_mapping_count": consecutive_invalid_count,
        "mapping_budget_complete": completed,
        "termination_reason": (
            "valid_mapping_target" if completed else "consecutive_invalid_limit"
        ),
        "elapsed_s": _positive_float(elapsed_s, "elapsed_s"),
        "mapping_sha256": _sha256(mapping_path),
        "processed_input_sha256": _sha256(processed_path),
        "output_subdirectory": output_dir.name,
        "front_end_projection_exact": True,
    }


def evaluate(input_path: Path, output_dir: Path) -> dict[str, object]:
    bundle = _validate_bundle(_load_json(input_path))
    assets = _verify_assets()
    output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/lib"
    from pytimeloop.timeloopfe import v4 as tl  # noqa: PLC0415

    os.chdir(ASSET_ROOT)
    layers = tuple(
        _evaluate_layer(
            manifest,
            bundle["layer_manifest_sha256"][ordinal],
            output_dir / f"medoid-{ordinal}",
            tl,
        )
        for ordinal, manifest in enumerate(bundle["layer_manifests"])
    )
    areas = tuple(item["area_square_meters"] for item in layers)
    if any(
        not math.isclose(areas[0], item, rel_tol=0.0, abs_tol=0.0) for item in areas
    ):
        raise RuntimeError("common architecture produced inconsistent layer areas")
    energy = sum(item["energy_joules"] * item["layer_multiplicity"] for item in layers)
    latency = sum(
        item["latency_seconds"] * item["layer_multiplicity"] for item in layers
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluator_id": EVALUATOR_ID,
        "candidate_sha256": bundle["candidate_sha256"],
        "compiled_plan_sha256": bundle["compiled_plan_sha256"],
        "panel_sha256": bundle["panel_sha256"],
        "objectives": {
            "energy_joules": _positive_float(energy, "aggregate energy"),
            "latency_seconds": _positive_float(latency, "aggregate latency"),
            "area_square_meters": areas[0],
        },
        "layers": list(layers),
        "protocol": bundle["protocol"],
        "provenance": {
            "asset_sha256": assets,
            "runner_sha256": _sha256(Path(__file__)),
            "python_hash_seed": "0",
            "mapper_seed_law": (
                "pinned_binary_random_pruned_default_constructed_cpp_engine"
            ),
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
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
    except Exception as error:  # process boundary publishes terminal evidence
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

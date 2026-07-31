"""Validate and expand the AgentEvolve cross-model experiment registry.

The registry is research orchestration, not optimizer behavior.  It lives
outside the library API and resolves workload-neutral model profiles from the
same composition objects used by live campaigns.  Reference blocks are fully
crossed.  Schema-v1 registries may additionally carry a mixed-level orthogonal
array; schema-v2 qualification registries explicitly cannot claim unimplemented
factor levels as executable cells.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from examples.development.corpus_paths import resolve_corpus_path


_AGENT_EVOLVE_ROOT = Path(__file__).resolve().parents[2]
if str(_AGENT_EVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_EVOLVE_ROOT))

from agent_evolve.campaign_presets import EQUAL_60_OFFSPRING_SCALE_SHAPES
from agent_evolve.integrations.pydantic_ai.model_execution_profile import (
    openrouter_model_execution_profile,
)
from examples.development.systematic_workload_contract import (
    WorkloadExecutionContract,
)


_TOKEN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_CELL_DOMAIN = b"agent-evolve:systematic-study-cell:v1\x00"
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2].parent


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_token(value: object, name: str) -> str:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed lowercase token grammar")
    return value


@dataclass(frozen=True, slots=True)
class StudyCell:
    """One immutable experiment cell expanded from a validated registry."""

    phase: str
    workload_id: str
    model_profile: str | None
    replicate_seed: int
    arm: str
    configuration: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _require_token(self.phase, "phase")
        _require_token(self.workload_id, "workload_id")
        if self.model_profile is not None:
            _require_token(self.model_profile, "model_profile")
        if type(self.replicate_seed) is not int or self.replicate_seed < 0:
            raise ValueError("replicate_seed must be a non-negative exact integer")
        if self.arm not in ("treatment", "control"):
            raise ValueError("arm must be treatment or control")
        if type(self.configuration) is not tuple or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not str
            for item in self.configuration
        ):
            raise TypeError("configuration must be an exact string-pair tuple")
        if self.configuration != tuple(sorted(set(self.configuration))):
            raise ValueError("configuration must be unique and canonical")
        if self.arm == "treatment" and self.model_profile is None:
            raise ValueError("treatment cells require a model profile")
        if self.arm == "control" and self.model_profile is not None:
            raise ValueError("shared controls cannot carry a model profile")

    @property
    def cell_sha256(self) -> str:
        return hashlib.sha256(
            _CELL_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()

    @property
    def cell_id(self) -> str:
        model = self.model_profile or "shared"
        return (
            f"{self.phase}_{self.workload_id}_{model}_{self.arm}_"
            f"s{self.replicate_seed}_{self.cell_sha256[:10]}"
        )

    def _unsigned_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "schema_version": 1,
            "phase": self.phase,
            "workload_id": self.workload_id,
            "model_profile": self.model_profile,
            "replicate_seed": self.replicate_seed,
            "arm": self.arm,
            "configuration": dict(self.configuration),
        }

    def to_record(self) -> dict[str, object]:
        return {
            **self._unsigned_record(),
            "cell_id": self.cell_id,
            "cell_sha256": self.cell_sha256,
        }


@dataclass(frozen=True, slots=True)
class SystematicExperimentStudy:
    """Validated registry plus deterministic cell expansion."""

    path: Path
    record: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "SystematicExperimentStudy":
        resolved = resolve_corpus_path(path).expanduser().resolve(strict=True)
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if type(value) is not dict:
            raise TypeError("study registry root must be an object")
        study = cls(path=resolved, record=value)
        study.validate()
        return study

    @property
    def study_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.record)).hexdigest()

    @property
    def model_names(self) -> tuple[str, ...]:
        return tuple(value["profile_name"] for value in self.record["models"])

    @property
    def workload_ids(self) -> tuple[str, ...]:
        return tuple(value["workload_id"] for value in self.record["workloads"])

    def _validate_models(self) -> None:
        models = self.record.get("models")
        if type(models) is not list or not models:
            raise ValueError("models must be a non-empty list")
        names: list[str] = []
        for value in models:
            if type(value) is not dict:
                raise TypeError("model entries must be objects")
            name = _require_token(value.get("profile_name"), "profile_name")
            profile = openrouter_model_execution_profile(name)
            if value.get("requested_model") != profile.requested_model:
                raise ValueError(f"registry model drift for {name}")
            if value.get("profile_sha256") != profile.profile_sha256:
                raise ValueError(f"registry profile hash drift for {name}")
            names.append(name)
        if names != sorted(set(names)):
            raise ValueError("models must be unique and canonical")

    def _validate_workloads(self) -> None:
        workloads = self.record.get("workloads")
        if type(workloads) is not list or not workloads:
            raise ValueError("workloads must be a non-empty list")
        identifiers: list[str] = []
        for value in workloads:
            if type(value) is not dict:
                raise TypeError("workload entries must be objects")
            identifier = _require_token(value.get("workload_id"), "workload_id")
            _require_token(value.get("domain"), "domain")
            contract_path = value.get("execution_contract")
            contract_sha256 = value.get("execution_contract_sha256")
            if contract_path is not None or contract_sha256 is not None:
                if type(contract_path) is not str or not re.fullmatch(
                    r"[0-9a-f]{64}", str(contract_sha256)
                ):
                    raise ValueError(f"invalid execution contract identity for {identifier}")
                WorkloadExecutionContract.load(
                    _WORKSPACE_ROOT / contract_path,
                    workspace_root=_WORKSPACE_ROOT,
                    expected_workload_id=identifier,
                    expected_sha256=str(contract_sha256),
                )
            else:
                runner = value.get("treatment_runner")
                if type(runner) is not str or not (_WORKSPACE_ROOT / runner).is_file():
                    raise ValueError(f"missing treatment runner for {identifier}")
                control_runner = value.get("control_runner")
                if (
                    type(control_runner) is not str
                    or not (_WORKSPACE_ROOT / control_runner).is_file()
                ):
                    raise ValueError(f"missing control runner for {identifier}")
            identifiers.append(identifier)
        if identifiers != sorted(set(identifiers)):
            raise ValueError("workloads must be unique and canonical")

    @staticmethod
    def _orthogonal(rows: list[list[int]], level_counts: tuple[int, ...]) -> bool:
        row_count = len(rows)
        for column, levels in enumerate(level_counts):
            expected = row_count // levels
            if any(
                sum(row[column] == level for row in rows) != expected
                for level in range(1, levels + 1)
            ):
                return False
        for left in range(len(level_counts)):
            for right in range(left + 1, len(level_counts)):
                expected = row_count // (level_counts[left] * level_counts[right])
                if any(
                    sum(
                        row[left] == left_level and row[right] == right_level
                        for row in rows
                    )
                    != expected
                    for left_level in range(1, level_counts[left] + 1)
                    for right_level in range(1, level_counts[right] + 1)
                ):
                    return False
        return True

    def _validate_factorial(self) -> None:
        factorial = self.record.get("mechanism_factorial")
        if type(factorial) is not dict:
            raise ValueError("mechanism_factorial must be an object")
        factors = factorial.get("factors")
        rows = factorial.get("orthogonal_array")
        if type(factors) is not list or type(rows) is not list:
            raise ValueError("factorial factors and rows must be lists")
        names: list[str] = []
        level_counts: list[int] = []
        for factor in factors:
            if type(factor) is not dict:
                raise TypeError("factor definitions must be objects")
            names.append(_require_token(factor.get("factor_id"), "factor_id"))
            levels = factor.get("levels")
            if type(levels) is not list or len(levels) not in (2, 3) or any(
                type(value) is not str or not value for value in levels
            ):
                raise ValueError("each factor requires two or three string levels")
            level_counts.append(len(levels))
        if names != sorted(set(names)):
            raise ValueError("factor IDs must be unique and canonical")
        if tuple(sorted(level_counts)) != (2, 3, 3, 3, 3, 3, 3, 3):
            raise ValueError("the L18 registry requires one binary and seven ternary factors")
        if (
            len(rows) != 18
            or any(
                type(row) is not list
                or len(row) != len(factors)
                or any(type(value) is not int for value in row)
                for row in rows
            )
            or len({tuple(row) for row in rows}) != len(rows)
        ):
            raise ValueError("orthogonal_array must contain 18 unique integer rows")
        if not self._orthogonal(rows, tuple(level_counts)):
            raise ValueError("mechanism design is not pairwise orthogonal")
        if factorial.get("planned_candidate_occurrences") != 62:
            raise ValueError("mechanism factorial must preserve the 62-occurrence envelope")
        scale_factors = tuple(
            factor for factor in factors if factor.get("factor_id") == "scale_shape"
        )
        if len(scale_factors) != 1:
            raise ValueError("mechanism factorial requires exactly one scale_shape factor")
        scale_levels = scale_factors[0]["levels"]
        if scale_levels != list(EQUAL_60_OFFSPRING_SCALE_SHAPES):
            raise ValueError("scale_shape levels drifted from executable presets")
        level_occurrences = scale_factors[0].get("level_candidate_occurrences")
        expected_occurrences = [
            shape.planned_offspring_occurrences + 2
            for shape in EQUAL_60_OFFSPRING_SCALE_SHAPES.values()
        ]
        if level_occurrences != expected_occurrences:
            raise ValueError("scale_shape occurrence claims drifted from executable presets")
        if any(
            shape.portfolio_width > 8
            for shape in EQUAL_60_OFFSPRING_SCALE_SHAPES.values()
        ):
            raise ValueError("scale_shape exceeds the authenticated selector width")
        model = factorial.get("primary_model_profile")
        if model not in self.model_names:
            raise ValueError("factorial primary model is not registered")

    def validate(self) -> None:
        schema_version = self.record.get("schema_version")
        if schema_version not in (1, 2, 3):
            raise ValueError("unsupported study registry schema")
        _require_token(self.record.get("study_id"), "study_id")
        self._validate_models()
        self._validate_workloads()
        if schema_version == 1:
            self._validate_factorial()
        elif (
            self.record.get("qualification_only") is not True
            or "mechanism_factorial" in self.record
        ):
            raise ValueError(
                "post-v1 registry must be an explicit qualification-only block"
            )
        reference = self.record.get("reference_block")
        if type(reference) is not dict:
            raise ValueError("reference_block must be an object")
        if reference.get("model_profiles") != list(self.model_names):
            raise ValueError("reference block must cross every registered model")
        if reference.get("workloads") != list(self.workload_ids):
            raise ValueError("reference block must cross every registered workload")
        seeds = reference.get("replicate_seeds")
        if type(seeds) is not list or not seeds or any(
            type(value) is not int or value < 0 for value in seeds
        ):
            raise ValueError("reference seeds must be non-negative exact integers")

    def reference_cells(self) -> tuple[StudyCell, ...]:
        self.validate()
        reference = self.record["reference_block"]
        config = tuple(sorted(reference["configuration"].items()))
        cells = [
            StudyCell(
                phase="reference",
                workload_id=workload,
                model_profile=model,
                replicate_seed=seed,
                arm="treatment",
                configuration=config,
            )
            for seed in reference["replicate_seeds"]
            for workload in reference["workloads"]
            for model in reference["model_profiles"]
        ]
        cells.extend(
            StudyCell(
                phase="reference",
                workload_id=workload,
                model_profile=None,
                replicate_seed=seed,
                arm="control",
                configuration=config,
            )
            for seed in reference["replicate_seeds"]
            for workload in reference["workloads"]
        )
        return tuple(cells)

    def factorial_cells(self) -> tuple[StudyCell, ...]:
        self.validate()
        if self.record["schema_version"] != 1:
            return ()
        factorial = self.record["mechanism_factorial"]
        factors = factorial["factors"]
        cells: list[StudyCell] = []
        for row_index, row in enumerate(factorial["orthogonal_array"], start=1):
            configuration = tuple(
                sorted(
                    (
                        factor["factor_id"],
                        factor["levels"][level - 1],
                    )
                    for factor, level in zip(factors, row, strict=True)
                )
            )
            configuration = tuple(
                sorted((*configuration, ("design_row", f"l18_{row_index:02d}")))
            )
            for workload in self.workload_ids:
                cells.append(
                    StudyCell(
                        phase="mechanism",
                        workload_id=workload,
                        model_profile=factorial["primary_model_profile"],
                        replicate_seed=factorial["replicate_seed"],
                        arm="treatment",
                        configuration=configuration,
                    )
                )
        return tuple(cells)

    def summary(self) -> dict[str, object]:
        reference = self.reference_cells()
        factorial = self.factorial_cells()
        return {
            "schema_version": self.record["schema_version"],
            "study_id": self.record["study_id"],
            "study_sha256": self.study_sha256,
            "model_count": len(self.model_names),
            "workload_count": len(self.workload_ids),
            "reference_treatment_cells": sum(
                value.arm == "treatment" for value in reference
            ),
            "reference_shared_control_cells": sum(
                value.arm == "control" for value in reference
            ),
            "mechanism_factorial_cells": len(factorial),
            "total_initial_cells": len(reference) + len(factorial),
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument(
        "--emit",
        choices=("summary", "reference", "factorial", "all"),
        default="summary",
    )
    args = parser.parse_args()
    study = SystematicExperimentStudy.load(args.registry)
    if args.emit == "summary":
        value: object = study.summary()
    elif args.emit == "reference":
        value = [cell.to_record() for cell in study.reference_cells()]
    elif args.emit == "factorial":
        value = [cell.to_record() for cell in study.factorial_cells()]
    else:
        value = {
            "summary": study.summary(),
            "reference": [cell.to_record() for cell in study.reference_cells()],
            "factorial": [cell.to_record() for cell in study.factorial_cells()],
        }
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

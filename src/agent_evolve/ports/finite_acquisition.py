"""Workload-neutral ask/tell boundary for finite black-box acquisition.

The core optimizer owns no feature encoding and no configuration materializer.
A workload adapter exposes a finite set of already legal candidate identities
and normalized feature rows.  An optional acquisition integration selects a
batch by identity; trusted adapter code maps those identities back to sealed
configurations.  This permits Bayesian, evolutionary, and learned acquisition
experts to share one boundary without importing workload schemas into the
application layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}$")
_REQUEST_DOMAIN = b"agent-evolve:finite-acquisition-request:v1\x00"
_DECISION_DOMAIN = b"agent-evolve:finite-acquisition-decision:v1\x00"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _require_token(value: str, *, name: str) -> None:
    if type(value) is not str or _TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must use the closed token grammar")


def _require_sha256(value: str, *, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _finite(value: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite canonical float")


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionObjective:
    """One raw objective and the fixed affine frame used by acquisition."""

    metric_id: str
    sense: str
    ideal: float
    reference: float

    def __post_init__(self) -> None:
        _require_token(self.metric_id, name="metric_id")
        if self.sense not in ("min", "max"):
            raise ValueError("sense must be 'min' or 'max'")
        _finite(self.ideal, name="ideal")
        _finite(self.reference, name="reference")
        if self.ideal == self.reference:
            raise ValueError("ideal and reference must be distinct")
        if self.sense == "min" and self.ideal >= self.reference:
            raise ValueError("minimization ideal must be below reference")
        if self.sense == "max" and self.ideal <= self.reference:
            raise ValueError("maximization ideal must be above reference")

    def maximize_value(self, raw_value: float) -> float:
        """Map the fixed reference to zero and the ideal to one."""

        self.__post_init__()
        _finite(raw_value, name="raw_value")
        if self.sense == "min":
            return (self.reference - raw_value) / (self.reference - self.ideal)
        return (raw_value - self.reference) / (self.ideal - self.reference)

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "metric_id": self.metric_id,
            "sense": self.sense,
            "ideal_hex": self.ideal.hex(),
            "reference_hex": self.reference.hex(),
        }


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionCandidate:
    """One legal candidate row; the configuration remains adapter-owned."""

    candidate_id: str
    configuration_sha256: str
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_token(self.candidate_id, name="candidate_id")
        _require_sha256(self.configuration_sha256, name="configuration_sha256")
        if type(self.features) is not tuple or not self.features:
            raise ValueError("features must be a non-empty exact tuple")
        for value in self.features:
            _finite(value, name="feature")
            if not 0.0 <= value <= 1.0:
                raise ValueError("features must lie in [0,1]")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "candidate_id": self.candidate_id,
            "configuration_sha256": self.configuration_sha256,
            "features_hex": [value.hex() for value in self.features],
        }


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionObservation:
    """One prior real-evaluator result in the same feature frame."""

    candidate_id: str
    configuration_sha256: str
    features: tuple[float, ...]
    objectives: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        FiniteAcquisitionCandidate(
            candidate_id=self.candidate_id,
            configuration_sha256=self.configuration_sha256,
            features=self.features,
        )
        if type(self.objectives) is not tuple or not self.objectives:
            raise ValueError("objectives must be a non-empty exact tuple")
        metric_ids: list[str] = []
        for metric_id, value in self.objectives:
            _require_token(metric_id, name="objective metric_id")
            _finite(value, name="objective value")
            metric_ids.append(metric_id)
        if tuple(metric_ids) != tuple(sorted(set(metric_ids))):
            raise ValueError("objectives must be unique and metric-sorted")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "candidate_id": self.candidate_id,
            "configuration_sha256": self.configuration_sha256,
            "features_hex": [value.hex() for value in self.features],
            "objectives": [
                {"metric_id": metric_id, "value_hex": value.hex()}
                for metric_id, value in self.objectives
            ],
        }


@dataclass(frozen=True, slots=True, eq=False)
class FiniteAcquisitionRequest:
    """Immutable finite-pool ask request at one authenticated cutoff."""

    campaign_scope_sha256: str
    cutoff_index: int
    batch_size: int
    seed: int
    objectives: tuple[FiniteAcquisitionObjective, ...]
    observations: tuple[FiniteAcquisitionObservation, ...]
    candidates: tuple[FiniteAcquisitionCandidate, ...]
    request_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        _require_sha256(self.campaign_scope_sha256, name="campaign_scope_sha256")
        if type(self.cutoff_index) is not int or self.cutoff_index < 1:
            raise ValueError("cutoff_index must be positive")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative exact integer")
        if type(self.objectives) is not tuple or len(self.objectives) < 2:
            raise ValueError("multi-objective acquisition requires at least two axes")
        for axis in self.objectives:
            if type(axis) is not FiniteAcquisitionObjective:
                raise TypeError("objectives must contain exact axes")
            axis.__post_init__()
        metric_ids = tuple(axis.metric_id for axis in self.objectives)
        if metric_ids != tuple(sorted(set(metric_ids))):
            raise ValueError("objective axes must be unique and metric-sorted")
        if type(self.observations) is not tuple or not self.observations:
            raise ValueError("observations must be a non-empty exact tuple")
        if type(self.candidates) is not tuple or len(self.candidates) < self.batch_size:
            raise ValueError("candidate pool cannot underfill the requested batch")
        dimensions: set[int] = set()
        observed_ids: set[str] = set()
        observed_hashes: set[str] = set()
        for observation in self.observations:
            if type(observation) is not FiniteAcquisitionObservation:
                raise TypeError("observations must contain exact rows")
            observation.__post_init__()
            if tuple(value[0] for value in observation.objectives) != metric_ids:
                raise ValueError("observation objective axes differ from request")
            dimensions.add(len(observation.features))
            if (
                observation.candidate_id in observed_ids
                or observation.configuration_sha256 in observed_hashes
            ):
                raise ValueError("observations must have unique IDs and configurations")
            observed_ids.add(observation.candidate_id)
            observed_hashes.add(observation.configuration_sha256)
        candidate_ids: set[str] = set()
        candidate_hashes: set[str] = set()
        candidate_features: set[tuple[float, ...]] = set()
        for candidate in self.candidates:
            if type(candidate) is not FiniteAcquisitionCandidate:
                raise TypeError("candidates must contain exact rows")
            candidate.__post_init__()
            dimensions.add(len(candidate.features))
            if (
                candidate.candidate_id in candidate_ids
                or candidate.configuration_sha256 in candidate_hashes
                or candidate.features in candidate_features
            ):
                raise ValueError("candidate IDs, configurations, and features must be unique")
            if candidate.configuration_sha256 in observed_hashes:
                raise ValueError("candidate pool repeats an evaluated configuration")
            candidate_ids.add(candidate.candidate_id)
            candidate_hashes.add(candidate.configuration_sha256)
            candidate_features.add(candidate.features)
        if len(dimensions) != 1:
            raise ValueError("all feature rows must share one dimension")
        computed = hashlib.sha256(
            _REQUEST_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.request_sha256 not in ("", computed):
            raise ValueError("request_sha256 does not authenticate the request")
        object.__setattr__(self, "request_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "campaign_scope_sha256": self.campaign_scope_sha256,
            "cutoff_index": self.cutoff_index,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "objectives": [axis.to_record() for axis in self.objectives],
            "observations": [value.to_record() for value in self.observations],
            "candidates": [value.to_record() for value in self.candidates],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "request_sha256": self.request_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FiniteAcquisitionRequest
            and self.request_sha256 == other.request_sha256
        )

    __hash__ = None


@dataclass(frozen=True, slots=True)
class FiniteAcquisitionSelection:
    candidate_id: str
    configuration_sha256: str
    acquisition_value: float

    def __post_init__(self) -> None:
        _require_token(self.candidate_id, name="candidate_id")
        _require_sha256(self.configuration_sha256, name="configuration_sha256")
        _finite(self.acquisition_value, name="acquisition_value")

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "candidate_id": self.candidate_id,
            "configuration_sha256": self.configuration_sha256,
            "acquisition_value_hex": self.acquisition_value.hex(),
        }


@dataclass(frozen=True, slots=True, eq=False)
class FiniteAcquisitionDecision:
    request_sha256: str
    policy_id: str
    policy_version: int
    policy_definition_sha256: str
    selected: tuple[FiniteAcquisitionSelection, ...]
    diagnostics: tuple[tuple[str, str], ...] = ()
    decision_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        _require_sha256(self.request_sha256, name="request_sha256")
        _require_token(self.policy_id, name="policy_id")
        if type(self.policy_version) is not int or self.policy_version < 1:
            raise ValueError("policy_version must be positive")
        _require_sha256(
            self.policy_definition_sha256,
            name="policy_definition_sha256",
        )
        if type(self.selected) is not tuple or not self.selected:
            raise ValueError("selected must be a non-empty exact tuple")
        for value in self.selected:
            if type(value) is not FiniteAcquisitionSelection:
                raise TypeError("selected must contain exact rows")
            value.__post_init__()
        if len({value.candidate_id for value in self.selected}) != len(self.selected):
            raise ValueError("selected candidates must be unique")
        if type(self.diagnostics) is not tuple:
            raise TypeError("diagnostics must be an exact tuple")
        diagnostic_keys: list[str] = []
        for key, value in self.diagnostics:
            _require_token(key, name="diagnostic key")
            if type(value) is not str or not value.isascii():
                raise ValueError("diagnostic values must be ASCII strings")
            diagnostic_keys.append(key)
        if tuple(diagnostic_keys) != tuple(sorted(set(diagnostic_keys))):
            raise ValueError("diagnostics must be unique and key-sorted")
        computed = hashlib.sha256(
            _DECISION_DOMAIN + _canonical_json(self._unsigned_record())
        ).hexdigest()
        if self.decision_sha256 not in ("", computed):
            raise ValueError("decision_sha256 does not authenticate the decision")
        object.__setattr__(self, "decision_sha256", computed)

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "request_sha256": self.request_sha256,
            "policy": {
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
                "definition_sha256": self.policy_definition_sha256,
            },
            "selected": [value.to_record() for value in self.selected],
            "diagnostics": [
                {"key": key, "value": value} for key, value in self.diagnostics
            ],
        }

    def to_record(self) -> dict[str, object]:
        self.__post_init__()
        return {**self._unsigned_record(), "decision_sha256": self.decision_sha256}

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is FiniteAcquisitionDecision
            and self.decision_sha256 == other.decision_sha256
        )

    __hash__ = None


@runtime_checkable
class FiniteAcquisitionPolicy(Protocol):
    """Select identities from a sealed legal pool using prior observations."""

    policy_id: str
    policy_version: int
    definition_sha256: str

    def select(self, request: FiniteAcquisitionRequest) -> FiniteAcquisitionDecision: ...


__all__ = [
    "FiniteAcquisitionCandidate",
    "FiniteAcquisitionDecision",
    "FiniteAcquisitionObjective",
    "FiniteAcquisitionObservation",
    "FiniteAcquisitionPolicy",
    "FiniteAcquisitionRequest",
    "FiniteAcquisitionSelection",
]

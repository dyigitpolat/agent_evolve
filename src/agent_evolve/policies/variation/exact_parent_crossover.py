"""Bounded exact two-parent crossover over immutable typed JSON.

This policy gives a model (or any other selector) only a finite set of opaque
locus handles.  The selector chooses which loci to import from the ordered
donor; it never authors candidate JSON.  Materialization starts from the exact
base parent and copies exact immutable donor subtrees, so all source
attribution is machine-derived.

The module is deliberately benchmark-, engine-, provider-, and framework-
neutral.  It has no synthesis, value witness, tolerance, or natural-language
rationale surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

from agent_evolve.domain.patch import (
    ArrayIndex,
    JsonPath,
    ObjectKey,
    canonical_path_bytes,
    require_sha256,
    validate_json_path,
)
from agent_evolve.domain.typed_json import (
    FrozenJsonArray,
    FrozenJsonObject,
    FrozenJsonValue,
    freeze_json,
    typed_json_equal,
    typed_json_sha256,
)


MIN_EXACT_PARENT_CROSSOVER_LOCI = 2
DEFAULT_MAX_EXACT_PARENT_CROSSOVER_LOCI = 256
MAX_EXACT_PARENT_CROSSOVER_LOCI = 4096

_POLICY = "bounded_exact_parent_import_v1"
_CONTRACT_DOMAIN = b"agent-evolve:exact-parent-crossover-contract:v1\x00"
_PLAN_DOMAIN = b"agent-evolve:exact-parent-crossover-plan:v1\x00"
_MATERIALIZATION_DOMAIN = b"agent-evolve:exact-parent-crossover-materialization:v1\x00"
_RECEIPT_DOMAIN = b"agent-evolve:exact-parent-crossover-receipt:v1\x00"
_EXCLUSIONS_DOMAIN = b"agent-evolve:exact-parent-import-exclusions:v1\x00"


def _canonical_json(record: dict[str, object]) -> bytes:
    return json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii", errors="strict")


def _hash_record(domain: bytes, record: dict[str, object]) -> str:
    return hashlib.sha256(domain + _canonical_json(record)).hexdigest()


def _validate_max_loci(max_loci: int) -> None:
    if type(max_loci) is not int:
        raise TypeError("max_loci must be an exact integer")
    if not (
        MIN_EXACT_PARENT_CROSSOVER_LOCI <= max_loci <= MAX_EXACT_PARENT_CROSSOVER_LOCI
    ):
        raise ValueError(
            "max_loci must lie in "
            f"[{MIN_EXACT_PARENT_CROSSOVER_LOCI}, "
            f"{MAX_EXACT_PARENT_CROSSOVER_LOCI}]"
        )


def _validate_parent(value: FrozenJsonObject, *, name: str) -> None:
    if type(value) is not FrozenJsonObject:
        raise TypeError(f"{name} must be an exact FrozenJsonObject")
    # Frozen dataclasses can still be reconstructed outside their generated
    # constructor.  Revalidate the complete graph at every public boundary.
    freeze_json(value)


def canonical_candidate_path_text(path: JsonPath) -> str:
    """Return an unambiguous candidate-relative display form for ``path``.

    Object keys always use JSON bracket notation.  This avoids collisions for
    keys containing dots, brackets, quotes, or decimal-looking text.
    Typed :class:`JsonPath` remains the authority; this string is a canonical
    wire/display projection of it.
    """

    validate_json_path(path)
    chunks = ["$"]
    for segment in path.segments:
        if type(segment) is ObjectKey:
            chunks.append(
                "["
                + json.dumps(
                    segment.value,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + "]"
            )
        elif type(segment) is ArrayIndex:
            chunks.append(f"[{segment.value}]")
        else:  # pragma: no cover - JsonPath validation closes the union.
            raise AssertionError("unsupported path segment")
    return "".join(chunks)


def _canonical_locus_id(ordinal: int) -> str:
    if type(ordinal) is not int or not (
        1 <= ordinal <= MAX_EXACT_PARENT_CROSSOVER_LOCI
    ):
        raise ValueError("locus ordinal is outside the policy bound")
    # IDs are compact contract-scoped handles.  The contract digest binds each
    # ordinal to its exact path and parent subtree hashes.
    return f"locus_{ordinal:04d}"


@dataclass(frozen=True, slots=True)
class ExactParentCrossoverLocus:
    """One nonoverlapping parent-discriminating locus."""

    locus_id: str
    path: JsonPath
    path_text: str
    base_value_sha256: str
    donor_value_sha256: str

    def __post_init__(self) -> None:
        if type(self.locus_id) is not str:
            raise TypeError("locus_id must be an exact string")
        validate_json_path(self.path)
        if self.path_text != canonical_candidate_path_text(self.path):
            raise ValueError("path_text is not the canonical projection of path")
        require_sha256(self.base_value_sha256, "base_value_sha256")
        require_sha256(self.donor_value_sha256, "donor_value_sha256")
        if self.base_value_sha256 == self.donor_value_sha256:
            raise ValueError("a crossover locus must discriminate the parents")

    def to_record(self) -> dict[str, object]:
        ExactParentCrossoverLocus.__post_init__(self)
        return {
            "locus_id": self.locus_id,
            "path_text": self.path_text,
            "path_schema_identity": self.path.schema_identity,
            "base_value_sha256": self.base_value_sha256,
            "donor_value_sha256": self.donor_value_sha256,
        }


def _validate_locus_tuple(
    loci: tuple[ExactParentCrossoverLocus, ...],
    *,
    max_loci: int,
) -> None:
    if type(loci) is not tuple or any(
        type(locus) is not ExactParentCrossoverLocus for locus in loci
    ):
        raise TypeError("loci must be an exact tuple of exact crossover loci")
    if not (MIN_EXACT_PARENT_CROSSOVER_LOCI <= len(loci) <= max_loci):
        raise ValueError(
            "a crossover contract requires at least two loci and may not "
            "exceed max_loci"
        )
    for locus in loci:
        ExactParentCrossoverLocus.__post_init__(locus)
    expected_ids = tuple(
        _canonical_locus_id(index) for index in range(1, len(loci) + 1)
    )
    if tuple(locus.locus_id for locus in loci) != expected_ids:
        raise ValueError("locus IDs must be the canonical contract-scoped sequence")
    path_keys = tuple(canonical_path_bytes(locus.path) for locus in loci)
    if path_keys != tuple(sorted(path_keys)):
        raise ValueError("loci must use canonical typed-path order")
    if len(set(path_keys)) != len(path_keys):
        raise ValueError("locus paths must be unique")
    terminal = object()
    root: dict[object, object] = {}
    for locus in loci:
        node = root
        for segment in locus.path.segments:
            if terminal in node:
                raise ValueError("locus paths must not overlap")
            child = node.setdefault(segment, {})
            if type(child) is not dict:  # pragma: no cover - local invariant.
                raise AssertionError("invalid locus path trie")
            node = child
        if node:
            # A prior path is either identical (already rejected above) or a
            # strict descendant of this locus.
            raise ValueError("locus paths must not overlap")
        node[terminal] = True


@dataclass(frozen=True, slots=True)
class ExactParentCrossoverContract:
    """Canonical finite action contract for ordered base and donor parents."""

    max_loci: int
    base_parent_sha256: str
    donor_parent_sha256: str
    loci: tuple[ExactParentCrossoverLocus, ...]

    def __post_init__(self) -> None:
        _validate_max_loci(self.max_loci)
        require_sha256(self.base_parent_sha256, "base_parent_sha256")
        require_sha256(self.donor_parent_sha256, "donor_parent_sha256")
        if self.base_parent_sha256 == self.donor_parent_sha256:
            raise ValueError("ordered crossover parents must be exact-distinct")
        _validate_locus_tuple(self.loci, max_loci=self.max_loci)

    def to_record(self) -> dict[str, object]:
        ExactParentCrossoverContract.__post_init__(self)
        return {
            "schema_version": 1,
            "policy": _POLICY,
            "max_loci": self.max_loci,
            "base_parent_sha256": self.base_parent_sha256,
            "donor_parent_sha256": self.donor_parent_sha256,
            "loci": [locus.to_record() for locus in self.loci],
        }

    @property
    def contract_sha256(self) -> str:
        return _hash_record(_CONTRACT_DOMAIN, self.to_record())


def _append_discriminating_loci(
    base: FrozenJsonValue,
    donor: FrozenJsonValue,
    path: JsonPath,
    loci: list[tuple[JsonPath, FrozenJsonValue, FrozenJsonValue]],
    *,
    max_loci: int,
) -> None:
    if typed_json_equal(base, donor):
        return

    if type(base) is FrozenJsonObject and type(donor) is FrozenJsonObject:
        base_keys = tuple(key for key, _ in base.items)
        donor_keys = tuple(key for key, _ in donor.items)
        if base_keys == donor_keys:
            for (key, base_item), (_, donor_item) in zip(
                base.items, donor.items, strict=True
            ):
                _append_discriminating_loci(
                    base_item,
                    donor_item,
                    path.child_key(key),
                    loci,
                    max_loci=max_loci,
                )
            return

    if type(base) is FrozenJsonArray and type(donor) is FrozenJsonArray:
        if len(base.items) == len(donor.items):
            for index, (base_item, donor_item) in enumerate(
                zip(base.items, donor.items, strict=True)
            ):
                _append_discriminating_loci(
                    base_item,
                    donor_item,
                    path.child_index(index),
                    loci,
                    max_loci=max_loci,
                )
            return

    # Scalars, type changes, object key-set changes, and array length changes
    # are indivisible exact imports at their nearest containing subtree.
    loci.append((path, base, donor))
    if len(loci) > max_loci:
        raise ValueError("derived crossover contract exceeds max_loci")


def derive_exact_parent_crossover_contract(
    *,
    base: FrozenJsonObject,
    donor: FrozenJsonObject,
    max_loci: int = DEFAULT_MAX_EXACT_PARENT_CROSSOVER_LOCI,
) -> ExactParentCrossoverContract:
    """Derive the canonical bounded locus frontier for ordered parents.

    Equal subtrees are omitted.  Same-shape objects and arrays are recursively
    split; a topology or runtime-type mismatch is represented by one exact
    containing-subtree locus.  Derivation fails closed when the exact frontier
    has fewer than two or more than ``max_loci`` members.
    """

    _validate_parent(base, name="base")
    _validate_parent(donor, name="donor")
    _validate_max_loci(max_loci)

    derived: list[tuple[JsonPath, FrozenJsonValue, FrozenJsonValue]] = []
    _append_discriminating_loci(
        base,
        donor,
        JsonPath(),
        derived,
        max_loci=max_loci,
    )
    derived.sort(key=lambda item: canonical_path_bytes(item[0]))
    if len(derived) < MIN_EXACT_PARENT_CROSSOVER_LOCI:
        raise ValueError(
            "ordered parents expose fewer than two exact discriminating loci"
        )

    loci = tuple(
        ExactParentCrossoverLocus(
            locus_id=_canonical_locus_id(index),
            path=path,
            path_text=canonical_candidate_path_text(path),
            base_value_sha256=typed_json_sha256(base_value),
            donor_value_sha256=typed_json_sha256(donor_value),
        )
        for index, (path, base_value, donor_value) in enumerate(derived, start=1)
    )
    return ExactParentCrossoverContract(
        max_loci=max_loci,
        base_parent_sha256=typed_json_sha256(base),
        donor_parent_sha256=typed_json_sha256(donor),
        loci=loci,
    )


@dataclass(frozen=True, slots=True)
class ExactParentImportPlan:
    """A canonical, bounded donor-locus selection for one exact contract."""

    contract_sha256: str
    locus_count: int
    import_locus_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sha256(self.contract_sha256, "contract_sha256")
        if type(self.locus_count) is not int:
            raise TypeError("locus_count must be an exact integer")
        if not (
            MIN_EXACT_PARENT_CROSSOVER_LOCI
            <= self.locus_count
            <= MAX_EXACT_PARENT_CROSSOVER_LOCI
        ):
            raise ValueError("locus_count is outside the crossover policy bound")
        if type(self.import_locus_ids) is not tuple or any(
            type(value) is not str for value in self.import_locus_ids
        ):
            raise TypeError("import_locus_ids must be an exact tuple of strings")
        if len(set(self.import_locus_ids)) != len(self.import_locus_ids):
            raise ValueError("import_locus_ids must be unique")
        if self.import_locus_ids != tuple(sorted(self.import_locus_ids)):
            raise ValueError("import_locus_ids must use canonical contract order")
        if not self.import_locus_ids:
            raise ValueError("a crossover plan must import at least one donor locus")
        if len(self.import_locus_ids) >= self.locus_count:
            raise ValueError("a crossover plan must retain at least one base locus")

    def to_record(self) -> dict[str, object]:
        ExactParentImportPlan.__post_init__(self)
        return {
            "schema_version": 1,
            "policy": _POLICY,
            "contract_sha256": self.contract_sha256,
            "locus_count": self.locus_count,
            "import_locus_ids": list(self.import_locus_ids),
        }

    @property
    def plan_sha256(self) -> str:
        return _hash_record(_PLAN_DOMAIN, self.to_record())


def build_exact_parent_import_plan(
    contract: ExactParentCrossoverContract,
    import_locus_ids: tuple[str, ...],
) -> ExactParentImportPlan:
    """Validate model-sized locus handles against their exact contract."""

    if type(contract) is not ExactParentCrossoverContract:
        raise TypeError("contract must be an exact ExactParentCrossoverContract")
    ExactParentCrossoverContract.__post_init__(contract)
    plan = ExactParentImportPlan(
        contract_sha256=contract.contract_sha256,
        locus_count=len(contract.loci),
        import_locus_ids=import_locus_ids,
    )
    allowed = {locus.locus_id for locus in contract.loci}
    if any(locus_id not in allowed for locus_id in plan.import_locus_ids):
        raise ValueError("import_locus_ids contains an ID outside the contract")
    selected = set(plan.import_locus_ids)
    contract_order = tuple(
        locus.locus_id for locus in contract.loci if locus.locus_id in selected
    )
    if plan.import_locus_ids != contract_order:
        raise ValueError("import_locus_ids must follow exact contract order")
    return plan


def validate_exact_parent_import_exclusions(
    contract: ExactParentCrossoverContract,
    forbidden_import_locus_sets: tuple[tuple[str, ...], ...],
) -> None:
    """Validate a canonical set of exact crossover actions to exclude.

    Every inner tuple is one otherwise-valid proper parent-import plan.  The
    outer tuple is lexicographically sorted and duplicate-free so prompt,
    schema, plan, and trace projections all bind one unambiguous value.
    """

    if type(contract) is not ExactParentCrossoverContract:
        raise TypeError("contract must be an exact ExactParentCrossoverContract")
    ExactParentCrossoverContract.__post_init__(contract)
    if type(forbidden_import_locus_sets) is not tuple or any(
        type(value) is not tuple for value in forbidden_import_locus_sets
    ):
        raise TypeError("forbidden import locus sets must be an exact tuple of tuples")
    for import_locus_ids in forbidden_import_locus_sets:
        build_exact_parent_import_plan(contract, import_locus_ids)
    if forbidden_import_locus_sets != tuple(sorted(set(forbidden_import_locus_sets))):
        raise ValueError(
            "forbidden import locus sets must be unique and canonically sorted"
        )
    if len(forbidden_import_locus_sets) == (1 << len(contract.loci)) - 2:
        raise ValueError(
            "forbidden import locus sets exhaust the exact crossover action space"
        )


def exact_parent_import_exclusions_sha256(
    contract: ExactParentCrossoverContract,
    forbidden_import_locus_sets: tuple[tuple[str, ...], ...],
) -> str:
    """Commit one contract-scoped canonical exclusion set."""

    validate_exact_parent_import_exclusions(contract, forbidden_import_locus_sets)
    return _hash_record(
        _EXCLUSIONS_DOMAIN,
        {
            "schema_version": 1,
            "policy": _POLICY,
            "contract_sha256": contract.contract_sha256,
            "forbidden_import_locus_sets": [
                list(value) for value in forbidden_import_locus_sets
            ],
        },
    )


class ExactParentSource(str, Enum):
    """Closed, machine-derived locus source vocabulary."""

    BASE = "base"
    DONOR = "donor"


@dataclass(frozen=True, slots=True)
class ExactParentLocusAttribution:
    """Hash-only evidence for one retained or imported exact subtree."""

    locus_id: str
    path_text: str
    source: ExactParentSource
    base_value_sha256: str
    donor_value_sha256: str
    source_value_sha256: str
    materialized_value_sha256: str

    def __post_init__(self) -> None:
        if type(self.locus_id) is not str or type(self.path_text) is not str:
            raise TypeError("attribution identifiers must be exact strings")
        if type(self.source) is not ExactParentSource:
            raise TypeError("attribution source must be an exact ExactParentSource")
        for name in (
            "base_value_sha256",
            "donor_value_sha256",
            "source_value_sha256",
            "materialized_value_sha256",
        ):
            require_sha256(getattr(self, name), name)
        expected = (
            self.base_value_sha256
            if self.source is ExactParentSource.BASE
            else self.donor_value_sha256
        )
        if self.source_value_sha256 != expected:
            raise ValueError("source hash does not match the attributed parent")
        if self.materialized_value_sha256 != self.source_value_sha256:
            raise ValueError("materialized locus does not match its exact source")

    def to_record(self) -> dict[str, object]:
        ExactParentLocusAttribution.__post_init__(self)
        return {
            "locus_id": self.locus_id,
            "path_text": self.path_text,
            "source": self.source.value,
            "base_value_sha256": self.base_value_sha256,
            "donor_value_sha256": self.donor_value_sha256,
            "source_value_sha256": self.source_value_sha256,
            "materialized_value_sha256": self.materialized_value_sha256,
        }


def _materialization_record(
    *,
    contract_sha256: str,
    plan_sha256: str,
    materialized_configuration_sha256: str,
    attributions: tuple[ExactParentLocusAttribution, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "policy": _POLICY,
        "contract_sha256": contract_sha256,
        "plan_sha256": plan_sha256,
        "materialized_configuration_sha256": (materialized_configuration_sha256),
        "attributions": [value.to_record() for value in attributions],
    }


def _value_at_path(root: FrozenJsonValue, path: JsonPath) -> FrozenJsonValue:
    validate_json_path(path)
    current = root
    missing = object()
    for segment in path.segments:
        if type(segment) is ObjectKey:
            if type(current) is not FrozenJsonObject:
                raise ValueError("object-key path traverses a non-object")
            match = next(
                (item for key, item in current.items if key == segment.value),
                missing,
            )
            if match is missing:
                raise ValueError("object-key path is absent")
            current = match  # type: ignore[assignment]
        elif type(segment) is ArrayIndex:
            if type(current) is not FrozenJsonArray:
                raise ValueError("array-index path traverses a non-array")
            if segment.value >= len(current.items):
                raise ValueError("array-index path is absent")
            current = current.items[segment.value]
        else:  # pragma: no cover - JsonPath validation closes the union.
            raise AssertionError("unsupported path segment")
    return current


def _replace_existing_path(
    root: FrozenJsonValue,
    path: JsonPath,
    replacement: FrozenJsonValue,
) -> FrozenJsonValue:
    if not path.segments:
        return replacement
    head = path.segments[0]
    tail = JsonPath(path.segments[1:])
    if type(head) is ObjectKey:
        if type(root) is not FrozenJsonObject:
            raise ValueError("object-key path traverses a non-object")
        found = False
        updated: list[tuple[str, FrozenJsonValue]] = []
        for key, item in root.items:
            if key == head.value:
                found = True
                updated.append((key, _replace_existing_path(item, tail, replacement)))
            else:
                updated.append((key, item))
        if not found:
            raise ValueError("object-key replacement path is absent")
        return FrozenJsonObject(tuple(updated))
    if type(head) is ArrayIndex:
        if type(root) is not FrozenJsonArray:
            raise ValueError("array-index path traverses a non-array")
        if head.value >= len(root.items):
            raise ValueError("array-index replacement path is absent")
        updated_items = list(root.items)
        updated_items[head.value] = _replace_existing_path(
            updated_items[head.value], tail, replacement
        )
        return FrozenJsonArray(tuple(updated_items))
    raise AssertionError("unsupported path segment")  # pragma: no cover


@dataclass(frozen=True, slots=True)
class ExactParentCrossoverReceipt:
    """Self-auditing hash receipt replayable with the exact ordered parents."""

    max_loci: int
    base_parent_sha256: str
    donor_parent_sha256: str
    contract_sha256: str
    plan_sha256: str
    import_locus_ids: tuple[str, ...]
    materialized_configuration_sha256: str
    materialization_sha256: str
    attributions: tuple[ExactParentLocusAttribution, ...]

    def __post_init__(self) -> None:
        _validate_max_loci(self.max_loci)
        for name in (
            "base_parent_sha256",
            "donor_parent_sha256",
            "contract_sha256",
            "plan_sha256",
            "materialized_configuration_sha256",
            "materialization_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if type(self.import_locus_ids) is not tuple or any(
            type(value) is not str for value in self.import_locus_ids
        ):
            raise TypeError(
                "receipt import_locus_ids must be an exact tuple of strings"
            )
        if type(self.attributions) is not tuple or any(
            type(value) is not ExactParentLocusAttribution
            for value in self.attributions
        ):
            raise TypeError("receipt attributions must be an exact tuple")
        for attribution in self.attributions:
            ExactParentLocusAttribution.__post_init__(attribution)
        if not (
            MIN_EXACT_PARENT_CROSSOVER_LOCI <= len(self.attributions) <= self.max_loci
        ):
            raise ValueError("receipt attribution count is outside its bound")
        if tuple(value.locus_id for value in self.attributions) != tuple(
            _canonical_locus_id(index) for index in range(1, len(self.attributions) + 1)
        ):
            raise ValueError("receipt attributions must use canonical locus order")
        plan = ExactParentImportPlan(
            contract_sha256=self.contract_sha256,
            locus_count=len(self.attributions),
            import_locus_ids=self.import_locus_ids,
        )
        if plan.plan_sha256 != self.plan_sha256:
            raise ValueError("receipt plan hash does not match its import plan")
        imported = set(self.import_locus_ids)
        if not imported <= {value.locus_id for value in self.attributions}:
            raise ValueError("receipt imports a locus outside its attribution set")
        for attribution in self.attributions:
            expected_source = (
                ExactParentSource.DONOR
                if attribution.locus_id in imported
                else ExactParentSource.BASE
            )
            if attribution.source is not expected_source:
                raise ValueError("receipt attribution contradicts its import plan")
        expected_materialization = _hash_record(
            _MATERIALIZATION_DOMAIN,
            _materialization_record(
                contract_sha256=self.contract_sha256,
                plan_sha256=self.plan_sha256,
                materialized_configuration_sha256=(
                    self.materialized_configuration_sha256
                ),
                attributions=self.attributions,
            ),
        )
        if self.materialization_sha256 != expected_materialization:
            raise ValueError("receipt materialization hash does not match its evidence")

    def to_record(self) -> dict[str, object]:
        ExactParentCrossoverReceipt.__post_init__(self)
        return {
            "schema_version": 1,
            "policy": _POLICY,
            "max_loci": self.max_loci,
            "base_parent_sha256": self.base_parent_sha256,
            "donor_parent_sha256": self.donor_parent_sha256,
            "contract_sha256": self.contract_sha256,
            "plan_sha256": self.plan_sha256,
            "import_locus_ids": list(self.import_locus_ids),
            "materialized_configuration_sha256": (
                self.materialized_configuration_sha256
            ),
            "materialization_sha256": self.materialization_sha256,
            "attributions": [value.to_record() for value in self.attributions],
        }

    @property
    def receipt_sha256(self) -> str:
        return _hash_record(_RECEIPT_DOMAIN, self.to_record())


@dataclass(frozen=True, slots=True)
class ExactParentCrossoverMaterialization:
    """Exact child, finite plan, exhaustive source evidence, and receipt."""

    configuration: FrozenJsonObject
    contract: ExactParentCrossoverContract
    plan: ExactParentImportPlan
    attributions: tuple[ExactParentLocusAttribution, ...]

    def __post_init__(self) -> None:
        _validate_parent(self.configuration, name="configuration")
        if type(self.contract) is not ExactParentCrossoverContract:
            raise TypeError("contract must be exact")
        ExactParentCrossoverContract.__post_init__(self.contract)
        if type(self.plan) is not ExactParentImportPlan:
            raise TypeError("plan must be exact")
        ExactParentImportPlan.__post_init__(self.plan)
        if self.plan.contract_sha256 != self.contract.contract_sha256:
            raise ValueError("plan does not bind the supplied contract")
        expected_plan = build_exact_parent_import_plan(
            self.contract, self.plan.import_locus_ids
        )
        if self.plan.to_record() != expected_plan.to_record():
            raise ValueError("plan is not canonical for the supplied contract")
        if type(self.attributions) is not tuple or any(
            type(value) is not ExactParentLocusAttribution
            for value in self.attributions
        ):
            raise TypeError("attributions must be an exact tuple")
        if len(self.attributions) != len(self.contract.loci):
            raise ValueError("source evidence must cover every contract locus")
        imported = set(self.plan.import_locus_ids)
        for locus, attribution in zip(
            self.contract.loci, self.attributions, strict=True
        ):
            ExactParentLocusAttribution.__post_init__(attribution)
            if (
                attribution.locus_id != locus.locus_id
                or attribution.path_text != locus.path_text
                or attribution.base_value_sha256 != locus.base_value_sha256
                or attribution.donor_value_sha256 != locus.donor_value_sha256
            ):
                raise ValueError("source evidence does not match its contract locus")
            expected_source = (
                ExactParentSource.DONOR
                if locus.locus_id in imported
                else ExactParentSource.BASE
            )
            if attribution.source is not expected_source:
                raise ValueError("source evidence contradicts the import plan")
            if (
                typed_json_sha256(_value_at_path(self.configuration, locus.path))
                != attribution.materialized_value_sha256
            ):
                raise ValueError("source evidence does not match the exact child")

    @property
    def materialized_configuration_sha256(self) -> str:
        return typed_json_sha256(self.configuration)

    def to_record(self) -> dict[str, object]:
        ExactParentCrossoverMaterialization.__post_init__(self)
        return _materialization_record(
            contract_sha256=self.contract.contract_sha256,
            plan_sha256=self.plan.plan_sha256,
            materialized_configuration_sha256=(self.materialized_configuration_sha256),
            attributions=self.attributions,
        )

    @property
    def materialization_sha256(self) -> str:
        return _hash_record(_MATERIALIZATION_DOMAIN, self.to_record())

    @property
    def receipt(self) -> ExactParentCrossoverReceipt:
        return ExactParentCrossoverReceipt(
            max_loci=self.contract.max_loci,
            base_parent_sha256=self.contract.base_parent_sha256,
            donor_parent_sha256=self.contract.donor_parent_sha256,
            contract_sha256=self.contract.contract_sha256,
            plan_sha256=self.plan.plan_sha256,
            import_locus_ids=self.plan.import_locus_ids,
            materialized_configuration_sha256=(self.materialized_configuration_sha256),
            materialization_sha256=self.materialization_sha256,
            attributions=self.attributions,
        )


def materialize_exact_parent_crossover(
    *,
    base: FrozenJsonObject,
    donor: FrozenJsonObject,
    contract: ExactParentCrossoverContract,
    import_locus_ids: tuple[str, ...],
) -> ExactParentCrossoverMaterialization:
    """Copy selected exact donor subtrees into the exact base parent."""

    _validate_parent(base, name="base")
    _validate_parent(donor, name="donor")
    if type(contract) is not ExactParentCrossoverContract:
        raise TypeError("contract must be an exact ExactParentCrossoverContract")
    ExactParentCrossoverContract.__post_init__(contract)

    # Never trust a detached or reconstructed contract.  Re-derive the full
    # exact frontier from the supplied ordered parents before materialization.
    expected_contract = derive_exact_parent_crossover_contract(
        base=base,
        donor=donor,
        max_loci=contract.max_loci,
    )
    if contract.to_record() != expected_contract.to_record():
        raise ValueError("contract does not match the supplied ordered parents")

    plan = build_exact_parent_import_plan(contract, import_locus_ids)
    imported = set(plan.import_locus_ids)
    child: FrozenJsonValue = base
    attributions: list[ExactParentLocusAttribution] = []
    for locus in contract.loci:
        source = (
            ExactParentSource.DONOR
            if locus.locus_id in imported
            else ExactParentSource.BASE
        )
        source_root = donor if source is ExactParentSource.DONOR else base
        source_value = _value_at_path(source_root, locus.path)
        if source is ExactParentSource.DONOR:
            child = _replace_existing_path(child, locus.path, source_value)
        materialized_value = _value_at_path(child, locus.path)
        attributions.append(
            ExactParentLocusAttribution(
                locus_id=locus.locus_id,
                path_text=locus.path_text,
                source=source,
                base_value_sha256=locus.base_value_sha256,
                donor_value_sha256=locus.donor_value_sha256,
                source_value_sha256=typed_json_sha256(source_value),
                materialized_value_sha256=typed_json_sha256(materialized_value),
            )
        )
    if type(child) is not FrozenJsonObject:  # Root imports cannot coexist with
        raise AssertionError("same-root object crossover produced a non-object")
    return ExactParentCrossoverMaterialization(
        configuration=child,
        contract=contract,
        plan=plan,
        attributions=tuple(attributions),
    )


def resolve_exact_parent_import_for_target(
    *,
    base: FrozenJsonObject,
    donor: FrozenJsonObject,
    contract: ExactParentCrossoverContract,
    target: FrozenJsonObject,
) -> tuple[str, ...] | None:
    """Resolve a known target to its exact donor-import action, if any.

    Resolution is linear in the bounded locus frontier: each target subtree
    must be typed-equal to exactly its base or donor counterpart.  A final
    materialization replay compares the complete target, which also proves
    that all shared/non-locus structure matches.  Parent identities are not
    actions in this policy (the import set must remain proper and nonempty), so
    they return ``None`` like every other unrepresentable target.
    """

    _validate_parent(base, name="base")
    _validate_parent(donor, name="donor")
    _validate_parent(target, name="target")
    if type(contract) is not ExactParentCrossoverContract:
        raise TypeError("contract must be an exact ExactParentCrossoverContract")
    ExactParentCrossoverContract.__post_init__(contract)
    expected_contract = derive_exact_parent_crossover_contract(
        base=base,
        donor=donor,
        max_loci=contract.max_loci,
    )
    if contract.to_record() != expected_contract.to_record():
        raise ValueError("contract does not match the supplied ordered parents")

    imported: list[str] = []
    try:
        for locus in contract.loci:
            target_value = _value_at_path(target, locus.path)
            base_value = _value_at_path(base, locus.path)
            donor_value = _value_at_path(donor, locus.path)
            if typed_json_equal(target_value, base_value):
                continue
            if typed_json_equal(target_value, donor_value):
                imported.append(locus.locus_id)
                continue
            return None
    except ValueError:
        # A topology-bearing target can make an otherwise valid parent path
        # absent or non-traversable.  Such a target is simply not representable.
        return None

    import_locus_ids = tuple(imported)
    if not import_locus_ids or len(import_locus_ids) == len(contract.loci):
        return None
    replayed = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=import_locus_ids,
    )
    if not typed_json_equal(replayed.configuration, target):
        return None
    return replayed.plan.import_locus_ids


def replay_exact_parent_crossover(
    *,
    base: FrozenJsonObject,
    donor: FrozenJsonObject,
    receipt: ExactParentCrossoverReceipt,
) -> ExactParentCrossoverMaterialization:
    """Re-derive, re-materialize, and exactly verify a prior receipt."""

    _validate_parent(base, name="base")
    _validate_parent(donor, name="donor")
    if type(receipt) is not ExactParentCrossoverReceipt:
        raise TypeError("receipt must be an exact ExactParentCrossoverReceipt")
    ExactParentCrossoverReceipt.__post_init__(receipt)
    contract = derive_exact_parent_crossover_contract(
        base=base,
        donor=donor,
        max_loci=receipt.max_loci,
    )
    if contract.contract_sha256 != receipt.contract_sha256:
        raise ValueError("receipt contract hash does not match the ordered parents")
    replayed = materialize_exact_parent_crossover(
        base=base,
        donor=donor,
        contract=contract,
        import_locus_ids=receipt.import_locus_ids,
    )
    if replayed.receipt.to_record() != receipt.to_record():
        raise ValueError("receipt does not match exact crossover replay")
    return replayed


__all__ = [
    "DEFAULT_MAX_EXACT_PARENT_CROSSOVER_LOCI",
    "MAX_EXACT_PARENT_CROSSOVER_LOCI",
    "MIN_EXACT_PARENT_CROSSOVER_LOCI",
    "ExactParentCrossoverContract",
    "ExactParentCrossoverLocus",
    "ExactParentCrossoverMaterialization",
    "ExactParentCrossoverReceipt",
    "ExactParentImportPlan",
    "ExactParentLocusAttribution",
    "ExactParentSource",
    "build_exact_parent_import_plan",
    "canonical_candidate_path_text",
    "derive_exact_parent_crossover_contract",
    "exact_parent_import_exclusions_sha256",
    "materialize_exact_parent_crossover",
    "replay_exact_parent_crossover",
    "resolve_exact_parent_import_for_target",
    "validate_exact_parent_import_exclusions",
]

# M4c lossless durable lineage codec

## Scope

M4c is an isolated outward adapter that gives the accepted M4b-v7 value graph
one lossless, versioned, canonical JSON representation. It has no event-store,
persistence, workflow, model, provider, evaluator, or benchmark capability.
Nothing in M4c wires the codec into a running system.

The implementation is
`agent_evolve.infrastructure.lineage_codec`. It depends only on the standard
library and the inward M4b domain/policy modules. The accepted M4b source,
tests, documentation, and independent artifact 49 remain unchanged.

## Public API

```python
from agent_evolve.infrastructure.lineage_codec import (
    LineageCodecError,
    LineageCodecLimits,
    decode_lineage_value,
    encode_lineage_value,
)

content = encode_lineage_value(value)
restored = decode_lineage_value(content)
assert encode_lineage_value(restored) == content
```

`encode_lineage_value` accepts only exact registered runtime types.
`decode_lineage_value` accepts only exact `bytes`. Both functions expose one
fixed, value-free failure message through `LineageCodecError`; rejected values,
parser errors, and hostile exceptions are not retained as a public cause or
context.

`LineageCodecLimits` bounds bytes, structural depth, nodes, container items,
individual UTF-8 strings, and integer digits. It is an external processing
policy rather than a serialized M4b value. The wire always includes all M4b
`TypedJsonLimits`, `PatchLimits`, patch schema, and envelope-version fields.

## Versioned canonical envelope

Every record has exactly three envelope fields:

```json
{
  "format": "agent_evolve.lineage_value",
  "schema_version": 1,
  "value": {"kind": "..."}
}
```

The displayed indentation is explanatory only. Admitted bytes use strict UTF-8
and JSON with no whitespace, ASCII-sorted field names, raw Unicode rather than
optional `\u` aliases, and no JSON floating-point tokens. Duplicate keys, a
UTF-8 BOM, invalid UTF-8, noncanonical field order, whitespace, unknown or
missing fields, unknown tags, aliases, and unsupported versions reject.

Before construction, the decoder:

1. applies the byte bound;
2. lexically bounds structural depth, nodes, and per-container items;
3. decodes strict UTF-8;
4. parses with duplicate-key, floating-token, and nonfinite-constant rejection;
5. validates exact built-in JSON container/scalar types and remaining bounds;
6. canonically re-encodes the parsed tree and requires byte equality;
7. decodes only the closed tag/field schema through public constructors;
8. revalidates authoritative aggregate values; and
9. encodes the reconstructed value again and requires byte equality.

There are no reflective constructors, dynamic imports, class names, pickle
payloads, default inference, or compatibility aliases.

## Closed value registry

The registry is an explicit tuple of exact type/tag pairs. Dispatch walks the
tuple with `is`, not dictionary lookup: even a rejected subclass with a hostile
metaclass cannot execute metaclass `__hash__` or `__eq__` during dispatch.

The complete 26 exported M4b dataclasses are supported:

- typed JSON: `TypedJsonLimits`, `FrozenJsonArray`, `FrozenJsonObject`;
- patch: `ObjectKey`, `ArrayIndex`, `JsonPath`, `PatchLimits`, all five patch
  operations, and `TypedPatch`;
- lineage: `CandidateOccurrence`, `VariationParent`, `ParentEdge`,
  `PreservationClaim`, `PreservationObligation`, and `VariationCase`; and
- variation policy: `ComponentTagAssignment`, `PatchRelation`,
  `ThreeWayPatchClassification`, `PreservationObligationRequest`,
  `PatchResolution`, `ParentConfiguration`, and
  `PreservationVerification`.

The transitive graph also has closed tags for `CandidateId`,
`OperatorInvocationId`, `InsightId`, `InsightRef`, all typed-JSON scalar types,
and every member of:

- `VariationKind`;
- `ParentRole`;
- `PreservationSource`;
- `PreservationExpectation`;
- `AbsenceContextKind`;
- `AbsenceFailureKind`;
- `ThreeWayRelationKind`; and
- `ResolutionChoice`.

These transitive types may also be encoded directly, which permits literal
goldens for every closed enum and scalar without fabricating an otherwise
invalid aggregate.

## Typed scalar and collection identity

Typed JSON uses dedicated records for `None`, Boolean, integer, finite float,
string, array, and object. Boolean is never accepted as integer. Integers use a
single canonical ASCII decimal spelling. Finite floats use exactly 16 lowercase
hex digits containing their IEEE-754 binary64 bits, so `0.0` and `-0.0`, as well
as adjacent finite values, retain bit identity across processes. NaN and
infinities are never admitted.

Strings and object keys remain exact strict UTF-8 text. Frozen object entries
are an ordered list of explicit key/value records, avoiding JSON object-key
coercion and preserving M4b's canonical UTF-8 key order. Tuple-valued domain
fields become ordered JSON arrays and reconstruct as exact tuples.

The codec does not preserve incidental Python object sharing: repeated immutable
value objects are reconstructed as equal validated values. M4b defines evidence
identity through explicit IDs, hashes, fields, and canonical projections rather
than `id(...)`, so this is lossless for the accepted value semantics.

## Complete aggregate fields

Encoding never omits a field because it equals a constructor default. In
particular:

- every typed-JSON limit is present;
- every patch limit and `TypedPatch.schema_version` is present;
- base/target/source candidate IDs and endpoint hashes are present;
- paths, indices, permutations, before/after values, component tags, and
  operation order are present;
- occurrence artifact/configuration hashes, sequence, and optional operator
  invocation are present;
- parent roles and tuple order are present;
- all absence-context receipt fields are present, using explicit JSON `null`
  only for absence;
- complete patches and relations remain inside three-way classifications;
- common ancestor, patches, selected insight versions, and obligations remain
  inside variation cases; and
- claims, resolutions, parent configurations, and preservation-verification
  projections retain every declared ID/hash field.

Two equal-content candidate occurrences with different `CandidateId` values
therefore remain distinct and in the same parent order after serialization.

## Exact validation before field use

The encoder first identifies a registered type with exact type identity. A
subclass, including one with overridden equality, hashing, string conversion,
iteration, truth testing, encoding, ordering, or a hostile metaclass, rejects
before any of those hooks can execute. An exact constructor-bypassing value is
accepted only if its complete authoritative validator succeeds. Malformed exact
values reject before hostile nested hooks.

### Transitive `InsightRef` validator-order debt

M4c discovered one upstream issue outside the frozen 26-class M4b audit:
`InsightRef.__post_init__` checks `isinstance(insight_id, InsightId)`. A forged
object can make `isinstance` execute a hostile `__class__` property before
`VariationCase` reaches its later exact-ID check.

M4c does not modify that accepted upstream file. Instead, its boundary performs
an exact, bounded prevalidation of `InsightRef`, `InsightId`, and positive exact
version fields before calling `InsightRef.__post_init__`. For `VariationCase`,
the complete exact `selected_insights` tuple is cardinality-bounded and
prevalidated before `validate_variation_case`. Direct and nested attack tests
observe zero `__class__` calls.

This is recorded as upstream validator-order debt. It does not weaken M4c's
codec boundary, but a future independently scoped domain cleanup should replace
the permissive upstream `isinstance` check with exact-type validation. The M4c
checkpoint pins `domain/ids.py` and `domain/insight.py` in addition to the M4b
files so that this containment remains auditable.

## Canonicality is not authentication

Canonical encoding detects noncanonical, malformed, mistagged, mistyped,
unknown-field, missing-field, invalid-constructor, and accidental corruptions
covered by the schema. It is not a signature or message-authentication code. A
party that deliberately rewrites a value into another valid record and
canonically re-encodes every affected field has authored a different valid
value. A future artifact/event layer must bind bytes to an external digest and
authority; M4c deliberately does not add that persistence or event mechanism.

## Verification coverage

The focused offline suite includes:

- an AST-derived inventory and exact round trip of all 26 dataclasses;
- every relevant ID, insight reference, enum member, operation, and resolution;
- scalar adjacency, signed zero, 1,000 random finite binary64 bit patterns,
  Unicode, and randomized nested typed trees;
- maximum domain limit fields, maximum index/integer/key/string cases, and a
  depth-64 typed tree;
- explicit nested limit and patch-schema wire assertions;
- equal-content/distinct-ID occurrence and parent-order preservation;
- literal goldens and subprocess hash-seed stability;
- one-byte, key, tag, type, field, tuple-order, and JSON field-order tampering;
- unknown versions/tags/aliases and operation retagging;
- duplicate keys, UTF-8/BOM/surrogate/noncanonical JSON, floating JSON tokens,
  and byte/depth/node/container/string/integer bounds;
- exact constructor bypass, invalid hostile fields, inherited/overriding
  subclasses, hostile metaclasses, and the direct/nested `InsightRef` attack;
- fixed public failures with no retained cause/context; and
- an AST dependency barrier against events, stores, application/session code,
  providers, evaluators, dynamic loading, reflection, and pickle.

The author checkpoint remains pending a fresh independent audit. Focused/full
counts, deterministic randomized-probe results, exact goldens, and file hashes
are frozen in research artifact 54 only after the final offline runs.

## Continuing holds

M4c does not authorize or implement:

- filesystem/object-store persistence;
- events, event schemas, journals, or replay projections;
- workflow/session/bootstrap composition;
- mutation, crossover, or recombination generation flows;
- model, provider, OpenRouter, DeepSeek, or OpenAI access;
- evaluators or benchmarks; or
- scientific/SOTA claims.

Those remain later milestones with separate design and independent-audit gates.

# M4b typed lineage and reversible patch foundation

Current status: **v7 repaired author checkpoint; fresh independent audit required**.

M4b's first implementation slice is an isolated, framework-free foundation for
reasoning about exact candidate ancestry. It does not call a model, evaluator,
event store, filesystem, or network service. It is not wired into the legacy
generation loop.

## Why a typed codec is necessary

Python and ordinary JSON tooling can conflate values that evolutionary evidence
must keep distinct. In particular, `True == 1 == 1.0` in Python. M4b therefore
uses an inward canonical codec for JSON-shaped configurations:

- accepted raw containers are exact built-in `dict` and `list` values;
- accepted scalars are exact `None`, `bool`, `int`, finite `float`, and `str`;
- objects and arrays freeze into immutable, runtime-distinct domain values;
- recursive validation rechecks exact tuple-backed storage and rejects forged
  mutable or cyclic frozen graphs;
- booleans, integers, floats, strings, arrays, and objects have distinct binary
  tags, with IEEE-754 float bytes preserving `0.0` versus `-0.0`;
- object keys use deterministic UTF-8 byte order; and
- cycles, container/scalar subclasses, lone surrogates, nonfinite floats, and
  values outside declared resource bounds fail closed.

Frozen arrays and objects use their validated canonical typed bytes for Python
equality, so `FrozenJsonArray((True,))` is not equal to arrays containing `1`
or `1.0`, and positive and negative floating-point zero remain distinct.
They are deliberately unhashable: evidence indexes must use explicit typed
digests rather than Python structural hashes.

The canonical bytes are a versioned hashing codec, not a replacement for a
human-facing JSON serializer or RFC 8785. Configuration hashes include the
`agent-evolve:typed-json:v1` domain tag.

## Patch algebra

A `TypedPatch` binds distinct base and target `CandidateId` occurrences, their
exact typed-configuration hashes, replay limits, schema version, and a canonical
tuple of operations. Paths use different value types for object keys and array
indices, so key `"0"` can never alias index `0`.

The closed operation vocabulary is:

- `replace_scalar` for exact scalar-to-scalar changes;
- `replace_subtree` for structural/type changes;
- `insert_sequence_item`;
- `delete_sequence_item`; and
- `permute_sequence`.

Every operation binds its typed path/schema identity, complete old and new
value state, derived old/new hashes, base occurrence, and optional semantic
component tag. Complete old/new state is intentional: hash-only records are not
invertible. Operations at equal or ancestor/descendant paths cannot coexist in
one patch. Application checks the global base hash, exact local old value,
operation result, exact local new value, and global target hash. Object paths
must already exist; application never creates a missing object key.

Patch-operation equality is validate-before-canonical and type-sensitive.
Operations and complete patches are unhashable; callers key them by explicit
operation bytes or `patch_hash`. Typed paths revalidate every leaf before
equality or hashing, so object-constructed path leaves cannot run hostile
equality/hash behavior through a trie or dictionary.

Derivation is deterministic. Equal-key objects recurse in canonical key order.
An object key-set change becomes an explicit subtree replacement. Arrays use a
dedicated operation for a pure permutation or exactly one insertion/deletion;
more complex structural edits become one explicit subtree replacement. For
duplicates, insertion/deletion chooses the lowest valid edit index and a
permutation maps each target element to the lowest still-unused equal source
index. This rule is type-sensitive and is tested exhaustively for small binary
sequences.

Typed JSON deliberately admits longer object keys than one bounded path
segment. When a changed key cannot be represented, derivation bubbles the edit
to the nearest representable parent subtree rather than failing or truncating
the key. Unchanged long-key siblings do not force a coarser diff, and tighter
patch-depth limits use the same conservative rule.

Patches and their inverses are canonical. Applying a patch and then its inverse
restores the exact typed base, including scalar-type and negative-zero identity.
Limits cap JSON depth, nodes, container entries, string bytes, integer bits,
canonical bytes, patch paths, operation count, and the exact framed patch-hash
preimage, including its fixed portion when an unchanged reproduction has zero
operations.

## Three-way classification, not merging

`classify_three_way_patches` freezes the exact ancestor and replays both branch
patches through their target postconditions before emitting a table. The bound
classification retains the ancestor. Its constructor and its resolution/query
consumers replay both patches and recompute the complete global relation
partition. Direct construction therefore cannot relabel effects, regroup an
invalidation, omit or duplicate an effect, or use a fabricated target hash.
Every consuming boundary recursively revalidates exact paths, limits,
operations, patches, relations, enum values, and resolutions. Relation
fingerprints include the operation's source occurrence, and classification
compares lossless per-side occurrence-count multisets rather than reducing
operations into a path-keyed dictionary.
Relation-level semantic-component tags use the same exact-string, nonempty,
strict-UTF-8, 256-byte validation as operation tags before any comparison,
hashing, or encoding. String subclasses and string-like objects therefore
cannot execute equality/encoding hooks or supply a different relation digest
preimage.
Tags whose character count already exceeds the byte cap are rejected before
UTF-8 encoding, avoiding an attacker-sized temporary allocation at this small
identity field.
The canonical table partitions every branch operation exactly once into:

1. identical effects at one path;
2. unmatched disjoint effects;
3. disjoint effects carrying the same explicit semantic-component tag;
4. different effects at the same path; or
5. strict ancestor/descendant overlaps where one edit invalidates the other's
   precondition.

The service does not synthesize a child or call an LLM. Conflict and invalidation
resolutions are separate immutable declarations. Validation requires one and
only one closed choice (`choose_left`, `choose_right`, `synthesize`, or
`drop_both`) for each required relation; synthesis carries only a promised
result hash at this boundary. Applying or judging a synthesized merge, and a
declared merge rule for compatible same-component groups, remain held.

## Occurrences, variation cases, and preservation

`CandidateOccurrence` separates proposal identity from content and artifact
hashes, so reproduction retains a new occurrence even when content is unchanged.
`ParentEdge` can exist only when its exact parent-to-child patch matches both
occurrence endpoints and points to a strictly later proposal sequence.
`VariationCase` rejects an input occurrence produced by the invocation being
constructed and binds ordered roles for reproduction,
mutation, crossover, three-way recombination, or repair; three-way cases require
an exact common ancestor and ordered ancestor-to-left/right patches.

This slice admits preservation obligations only for replay-verified three-way
cases. A request identifies a relation, a branch or neutral-identical source,
and a path contained in exactly one operation effect. The factory binds each
immutable obligation ID to the branch patch hash, operation-effect hash,
relation identity, exact path, and presence-aware ancestor and expected states.
Branch-specific obligations must differ from both the ancestor and the other
branch. Identical edits produce one neutral obligation and never count as use
of either parent. Explicit absence expectations make deletions verifiable.
An absence obligation additionally binds the exact failure location, surviving
container kind, missing-key/out-of-bounds reason, and a shape hash over object
keys or array length. Verification requires the same structural receipt in the
child, so replacing an ancestor container by a scalar or reshaping it cannot
masquerade as preservation of a leaf deletion.

Child claims contain only obligation IDs. Verification replays the
classification, re-derives every obligation, requires exact coverage, and
checks an exact child value or absence. An ancestor copy, one-parent copy,
unchanged ancestor content, fabricated obligation, missing claim, or altered
path cannot earn two-branch recombination credit. The corresponding
replay-derived contribution factory for two-parent crossover without an
ancestor remains held; manual obligations for that weaker operator are rejected.

The returned `PreservationVerification` projection recursively revalidates
every claim and candidate ID, requires 2–4,096 canonical unique claims, and
requires exactly two distinct used parent IDs. It is intentionally
non-authoritative by itself: future durable evidence must bind the projection
to the exact variation case, classification, child, and claims rather than
treating a structurally valid in-memory value as proof that verification ran.

Every exported M4b dataclass now uses validate-before-canonical equality.
Evidence aggregates, including occurrences, parents, cases, obligations,
requests, resolutions, bindings, relations, classifications, claims, and
verification projections, are deliberately unhashable; their explicit IDs or
canonical digests are the supported evidence keys. The two immutable limit
values and typed paths/path leaves remain hashable only through fully validated,
type-safe projections. An inherited equality implementation requires both its
receiver and operand to have the exact exported runtime type. An unrelated
foreign operand receives `False` directly when the exact M4b value owns Python
dispatch; consuming evidence boundaries never rely on equality against a
foreign or user-overridden subclass and always require exact runtime types.

Obligations, claims, selected insights, and component assignments have explicit
hard bounds. Patch-local overlap, cross-branch prefix classification, relation
validation, obligation lookup, and parent-local obligation overlap use tries
linear in total path length plus emitted overlap edges rather than pairwise
scans.

## Implemented files and author verification

The authoritative slice is:

- `src/agent_evolve/domain/typed_json.py` — immutable typed values and canonical
  hashing;
- `src/agent_evolve/domain/patch.py` — paths, limits, five operations, and patch
  value objects;
- `src/agent_evolve/domain/lineage.py` — occurrence, parent, variation-case,
  obligation, and claim values;
- `src/agent_evolve/policies/variation/typed_patch.py` — derive/apply/invert,
  verified three-way classification, resolution validation, and preservation
  verification; and
- `tests/test_lineage_patch_m4b.py` — deterministic property-style and
  adversarial kill tests.

The v7 author-focused checkpoint passes 64/64 tests. It includes 729 ordered
small-tree endpoint pairs, 900 requested length-0-to-5 duplicate binary insertion
and deletion transformations, inverse round trips, every operation and three-way category,
stale and forged records, canonical ordering, dependency direction, hostile
values, exact empty/single-operation byte boundaries, long-key bubbling,
resource bounds, replay-bound global classification, presence-aware deletion
obligations, ancestor-copy rejection, resolution completeness, and
ignored-parent controls. It additionally rejects hostile relation component
equality/encoding/hash objects at every public consumer, recursively rejects
forged or noncanonical preservation receipts, distinguishes typed values and
operations under implicit equality, rejects mutable/cyclic frozen graphs, and
proves zero hostile-hook calls across equality, hashing, set, dictionary, path,
limit, collection-bound, foreign-reflection, and exact-Boolean reproducers. The
complete offline AgentEvolve suite passes 357/357 tests with one unrelated
Pydantic-AI deprecation warning. A separate seeded v7 probe (`20260713`) passed 1,000 random
derive/apply/invert/determinism round trips and 250 random three-way partition
replays. A v7 scaling probe observed exact linear work counts through 2,048
operations and local median time ratios 2.010, 2.021, and 2.009 across successive
doublings. The fresh auditor must rerun these probes. An
independent read-only re-audit is still required before this repaired isolated
slice is called GO.

## Integration holds

This slice deliberately has no event types, artifact serializer/decoder, lineage
replay projection, checkpoint/resume behavior, workflow commands, operator
scheduler, complementarity policy, generator prompt/response schema, domain
schema invariant adapter, evaluator evidence, or M4a assignment decision. It
does not implement a three-way merge, LLM mutation, crossover, or repair. It
also does not authorize a provider, model, evaluator, benchmark, or `.env` call.

The exact contribution-evidence factory for two-parent crossover without a
common ancestor remains held. The implemented obligation factory is specific
to replayed ancestor-to-branch effects.

Before workflow integration, durable codecs must preserve every typed value,
limit, hash, relation, resolution, obligation, and claim losslessly; event replay
must prove occurrence uniqueness and patch reconstruction; synthetic generators
must fail ignored-parent controls; and compatible-component merge rules must be
explicit, deterministic where claimed, and independently audited.

## Superseded v2 repaired author-checkpoint identities

| Item | SHA-256 |
|---|---|
| typed JSON domain module | `b45a57a78f8d4f825d7c36e1db0f15597530b196978f2351e5afad60b4431f34` |
| patch domain module | `b71f8702de5153e7cadff3a2fa484ef700f027d1a940e1a9591443da43aff236` |
| lineage domain module | `ff9c5916bbdef83aa0767d419715eead7308b7c02ba045a0202de90ccbfd25e9` |
| variation policy module | `63601cc976071e20a99ffed1a38864239ca4a21c315d02823a21be9e173f76fd` |
| focused tests | `b684454a00b37d81415dc96e9055a01b730876f01902c5517c298a77482693a1` |

These are repaired author-checkpoint hashes, not an independent GO. Any further
repair changes them and requires rerunning the focused suite and updating the
evidence artifact.

## V3 repair checkpoint (audit pending)

The v3 repair closes the independently reproduced source-identity,
same-path-collapse, non-enum relation/resolution, mistyped-operation,
absence-context, recursive-limit/path validation, and quadratic-prefix-scan
failures. Each reproducer is now a focused regression. Deterministic complexity
tests observe 512 work units for 512 one-segment patch paths, 1,024 work units
for two disjoint 512-operation branches, and at most 1,024 units for a
512-edge root/descendant star.

| Item | V3 SHA-256 |
|---|---|
| typed JSON domain module | `35b08a8da9263fe652b76670e77fb20c4205abb489f6be9765d792c93aae5d08` |
| patch domain module | `0105b56925a0bfbd5dbf660c2924b7e75ab0a776fa46e651b6c05c763e838ffd` |
| lineage domain module | `3e71c59a1e3e849e4e55207fa8a813bd7b92adda3ed3d7aacd1c9ba430a41cf3` |
| typed variation policy | `f607b5d7b993491b7d6ffc40a7b09482a810b3f56dd2ec2d431d53610774a93d` |
| focused tests | `dbeda7dd5ec9b6472bb790ddb2196347f88cacecdb725088c27843a9b5bab5f4` |

These are author-produced v3 bytes, not an independent GO. Durable codecs,
events, workflow integration, models, providers, and evaluators remain held.

## V4 repair checkpoint (audit pending)

The v4 repair closes the independently reproduced hostile relation-component
identity and shallow preservation-receipt defects. Exact component validation
now precedes equality/encoding at both operation and relation boundaries.
`PreservationVerification` recursively validates nested claims and candidate
IDs, bounds and canonicalizes claim identities, and requires two distinct
three-way parent identities. A public recursive validator exists for future
consumers, while the projection remains explicitly non-authoritative until a
durable evidence graph binds its complete factory context.

| Item | V4 SHA-256 |
|---|---|
| typed JSON domain module | `35b08a8da9263fe652b76670e77fb20c4205abb489f6be9765d792c93aae5d08` |
| patch domain module | `03bf84ce45c6d2bd31697cc52fc2979b35166fdc72afd883dc1ce94142c14170` |
| lineage domain module | `3e71c59a1e3e849e4e55207fa8a813bd7b92adda3ed3d7aacd1c9ba430a41cf3` |
| typed variation policy | `c4d4ce5d359b02bd376da517122e398c7a68b7d497898e9b4484d08ac119f84b` |
| focused tests | `95b5d7f5f7313747281660142c33e0d27f8489d327c201a622af0f53c88ebb1b` |

These are author-produced v4 bytes, not an independent GO. The hostile
reproducers, randomized round trips/classifications, scaling probes, complete
suite, and exact hashes require a different agent's read-only audit. Durable
codecs, events, workflow integration, models, providers, and evaluators remain
held.

## V5 repair checkpoint (independently rejected)

The independent v4 audit in research artifact 43 found that generated
dataclass equality/hash still bypassed explicit validators, valid
boolean/integer/float operations inherited Python's untyped numeric equality,
recursive frozen validation accepted object-forged list storage, and several
constructors compared, hashed, encoded, or scanned nested values before
recursive validation and bounds.

V5 applies one evidence-value policy: frozen typed values and operations compare
through validated canonical bytes; identity-bearing aggregates compare through
validated canonical projections; all are unhashable when an explicit digest or
ID is the supported key. Recursive frozen validation requires exact tuple
storage and detects forged cycles. Paths validate leaves before implicit use.
Sequence operations validate complete frozen operands before length, slicing,
or permutation work. Source IDs, patch limits, verification limits, parent
tuples, relation-operation tuples, claims, and other bounded collections are
validated and capped before equality, hashing, sets, or proportional scans.

These author-produced v5 bytes were independently rejected in research artifact
44. Durable codecs, events, workflow integration, models, providers, evaluators,
and benchmark use remained held.

## V6 repair checkpoint (independently rejected)

The v5 audit found two remaining public-boundary defects. Generated dataclass
equality and hashing still let object-constructed invalid occurrences, parents,
cases, obligations, limits, assignments, requests, resolutions, and parent
bindings execute nested hooks or alias valid evidence. Separately,
`operation_effect_bytes(include_component=...)` accepted integers and arbitrary
truthy objects, allowing executable or nondeterministic identity input.

V6 inventories every exported M4b dataclass. Each now validates both exact
operands before comparing a type-safe projection. Evidence aggregates are
explicitly unhashable; typed JSON/patch limits retain validated hashes because
they are immutable value defaults. Existing custom equality methods return
`False` for foreign exact types rather than delegating to reflected equality.
The operation-effect encoder rejects anything whose exact type is not `bool`
before evaluating the flag.

Three new adversarial tests cover valid constructor-bypassing copies, invalid
nested fields, equality in both directions, hash/set/dictionary use, foreign
reflected equality, exact `True`/`False`, integers, and toggling truth objects.
All invalid exact-class paths reject with zero hostile equality, hash, Boolean,
encoding, or ordering hooks. Research artifact 46 independently rejected these
v6 bytes for the inherited-subclass dispatch gap below. Durable codecs, events,
workflow integration, models, providers, evaluators, benchmarks, and `.env`
access remained held.

## V7 repair checkpoint (audit pending)

Python gives a right-hand proper subclass priority for reflected equality. V6
checked only the operand type, so an inheriting subclass with valid copied
fields could run the base implementation as receiver and compare equal to the
exact base value. Fifteen of 26 exported dataclasses equality-aliased this way;
`TypedJsonLimits`, `PatchLimits`, `ObjectKey`, and `ArrayIndex` also produced the
same hash and collapsed in sets/dictionaries.

V7 requires an exact receiver as well as an exact operand in every custom
equality implementation. Validating hash helpers likewise reject a non-exact
receiver, and subclass access to preservation-obligation identity or
classification revalidation fails closed. A new test constructs an inheriting
subclass for all 26 exported M4b dataclasses, copies every valid field without a
constructor, and exercises both operand orders, list/tuple membership, hashing, sets, dictionaries,
obligation identity, and classification behavior. Every inherited-subclass
comparison returns `False`, and every attempted subclass hash rejects.

These are author-produced v7 bytes, not an independent GO. Durable codecs,
events, workflow integration, models, providers, evaluators, benchmarks, and
`.env` access remain held until a different agent attacks the pinned checkpoint.

# M1b content-addressed artifact store

This additive slice provides a framework-neutral byte artifact port plus verified
in-memory and filesystem adapters. An `ArtifactId` is `artifact_` followed by a
domain-separated SHA-256 identity digest over the exact media type and exact payload
bytes. `ArtifactRef.sha256_hex` separately remains the ordinary SHA-256 checksum of
only the payload bytes. `ArtifactRef` binds those values to immutable byte length and
exact media-type metadata. JSON helpers use sorted, whitespace-free,
finite-number-only UTF-8 JSON; text helpers use strict UTF-8.

The portable v2 identity preimage is:

```text
"agent-evolve:artifact-id:v2\0"
|| uint64_be(len(media_type_ascii)) || media_type_ascii
|| uint64_be(len(payload))          || payload
```

The `\0` shown in the domain tag is one NUL byte (`0x00`), lengths are byte
lengths, and `||` is byte concatenation. Media-type validation
restricts the label to printable ASCII. The versioned domain tag and explicit
framing prevent ambiguous concatenations. Identical bytes stored as two exact media
types therefore receive two distinct IDs, independent of insertion order, while
retaining the same raw-payload checksum.

`ArtifactId` enforces the `artifact_` plus lowercase 64-hex shape wherever it is
used, including event references. The value object alone cannot prove which bytes a
digest names; `ArtifactRef` construction and store verification enforce the binding
to media type, payload checksum, size, and payload bytes. Verifying that an event's
reference was durably registered belongs to the later event/artifact integration
boundary.

The filesystem adapter commits one self-describing file with a file `fsync`, atomic
rename, and directory `fsync`. This makes payload and metadata visible together.
Handled exceptions during payload write, payload `fsync`, or rename leave no target
artifact and clean the temporary file. Abrupt process termination can leave an
ignored `.artifact-write-*.tmp`; safe orphan reclamation is pending. A
directory-`fsync` failure happens after rename, so
the valid target may be visible while crash durability is uncertain; the exception
is preserved, and retrying the same `put_bytes` re-`fsync`s the root before returning
the existing reference. Thus an idempotent retry is also the recovery operation.
Every read verifies the byte count and digest, while construction verifies every
existing artifact file. Rewriting the same bytes with the same media type is
idempotent. Identity collisions, wrong expected media types, missing IDs, malformed
metadata, and corrupt payloads have separate errors. The on-disk container magic and
schema are version 2; version-1 files from the pre-hardening prototype are rejected
rather than silently reinterpreted because their IDs had different semantics.

## Trust boundary and deliberate limits

- The store persists exactly what its caller supplies. **Secret detection,
  redaction, and minimization are the caller/interceptor's responsibility.** Prompt,
  response, exception, environment, header, and trace interceptors must remove API
  keys, credentials, personal data, and other restricted material before calling
  `put_bytes`, `put_json`, or `put_text`.
- Content integrity is not authenticity. This slice intentionally adds no manifest,
  hash chain, signature, encryption, access-control layer, or remote replication.
  The typed identity makes a media-type or payload edit detectable under an existing
  reference, but it does not establish who produced that reference. Experiment
  manifests can bind references later without changing the byte-store port.
- The filesystem root is a dedicated flat directory. The implementation verifies
  complete payloads in memory and is suitable for pre-pilot prompts, responses,
  candidates, and diagnostics; streaming and sharded layouts are later scalability
  work. A configurable maximum artifact-size policy is intentionally pending; the
  byte-store port has not been burdened with a deployment policy before the active
  integration chooses limits.
- The exact media-type string is semantic identity and is not normalized. Identical
  bytes may be stored under multiple types, with a distinct ID for every exact type.
- Canonical JSON is the deliberately narrow AgentEvolve codec, not an implementation
  of RFC 8785. It supports ordinary JSON values, rejects non-string object keys,
  duplicate decoded keys, non-finite numbers, and non-standard numeric constants,
  rejects strings and keys that are not strictly UTF-8 encodable (including Python
  surrogate code points), and performs no Unicode normalization.
- The parent of a new store root must already exist. Creating the root is a
  single-directory operation and the parent directory is `fsync`ed on POSIX before
  construction returns. Artifact commits `fsync` the file, atomically rename it,
  and then `fsync` the root directory.
- All store instances for the same canonical root share a process-global lock.
  POSIX deployments additionally use `fcntl.flock` across processes. If importing
  `fcntl` is impossible, concurrent access remains safe only among threads and store
  instances in the current process; callers must ensure a single process owns that
  root. A runtime `flock` failure is fail-closed rather than silently weakening the
  guarantee. Directory `fsync` and cross-process locking remain POSIX-specific.
- This slice is not wired into the generation loop, evaluator, event recorder, or
  public optimization API. That integration requires explicit redaction and event
  boundary decisions.

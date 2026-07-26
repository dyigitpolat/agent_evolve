# M1 event core: first additive slice

This slice provides the immutable event primitives used by the target architecture:
runtime-distinct IDs, frozen payloads and envelopes, canonical schema-versioned JSON,
injectable clocks/ID factories, in-memory and append-only JSONL stores, an
append-before-publish recorder, and the version-1 `RunCounters` projection.

It is intentionally **not connected to the legacy generation loop yet**. Existing
`optimize`, `run_evolution_loop`, `on_event`, and `SearchResult.evaluations`
behavior therefore remains unchanged by the event slice. The next integration slice
must emit these events at proposal, validation, call, evaluation, cache, and admission
boundaries.

## Deliberate first-slice limits

- The JSONL API is append-only and validates schema, run identity, globally unique
  event IDs, contiguous sequences, monotonic offsets, and causation links to an
  earlier event in the same run. It is not yet a
  cryptographically hash-chained log; well-formed external edits require a later
  manifest/log digest to detect.
- Large-value artifact storage is not part of this slice. Payloads already use
  `ArtifactId` references so mutable candidate/prompt/response dictionaries do not
  enter the durable schema.
- `EvaluationCacheHit` represents a validated successful objective outcome only.
  Typed cached candidate failures can be added with a new payload/event schema when
  the cache policy is implemented; infrastructure/system failures must never be
  cached.
- `RunCounters` exposes generic evaluation totals plus an explicit `full` subset.
  Per-fidelity/seed breakdowns should be a projection-version addition, not mutable
  maps inside events.
- A recorder reopened in another process preserves sequence and strictly increasing
  monotonic offsets, but cannot reconstruct the old process's monotonic origin.
  Cross-process critical-path time and crash resume belong to the M2 workflow and
  checkpoint work.
- Observer exceptions propagate only after the event is durable. This makes observer
  failure visible without losing or reusing an event sequence; policy for optional
  exporter failures will be chosen when exporters are integrated.
- An append adapter can raise after making an event readable (for example, if a
  post-write directory fsync fails). `EventRecorder` reconciles the exact event at
  the stream tail, advances its local sequence without publishing sinks, and raises
  `EventAppendObservedError`. An unreadable or conflicting result raises
  `EventAppendReconciliationError`; callers must reconstruct/recover rather than
  retrying an ambiguous sequence blindly.
- JSONL currently revalidates all streams on each operation. This is correctness-first
  behavior for the small pre-pilot logs; a verified index/checkpoint can optimize it
  later without becoming authoritative.
- Failure events carry a required, redacted human-readable summary plus optional
  exception type and content-addressed diagnostics reference. Raw exception text,
  credentials, provider headers, and large stdout/stderr blobs must not be placed in
  the summary; the future artifact writer owns sanitization and durable blob storage.
- `evaluation_attempts_completed` counts both successful and failed physical attempts.
  A pre-start failure without an `EvaluationAttemptId` is classified but cannot close a
  nonexistent attempt. This keeps started/completed/failure reconciliation exact.

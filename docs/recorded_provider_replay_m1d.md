# M1d sanitized request capture and recorded-provider replay

Status: offline, provider-free vertical slice. It does not import or invoke a
provider SDK, HTTP client, model, or `.env`; it does not invoke or operationally
depend on the active optimization loop or evaluator.

## Additive event and artifact contract

M1d adds `ArtifactRole.LLM_REQUEST` and the frozen event
`LLMRequestArtifactLinked(call_id, request_artifact_id)`. This is additive to
event schema version 1; no existing payload shape or serialized field was
changed. A replayable logical call has this order:

1. journal a minimized/sanitized canonical JSON request as `llm_request`;
2. append `LLMCallRequested` containing only safe operation/model metadata;
3. append `LLMRequestArtifactLinked` referencing the preceding registration;
4. append `LLMCallStarted` and zero or more provider-attempt events;
5. journal a sanitized `llm_response` after a successful attempt, with the
   registration envelope caused by that exact `ProviderAttemptCompleted` event;
6. append `LLMCallCompleted` referencing that preceding response registration,
   or append a bounded-summary `LLMCallFailed`.

The request JSON may contain messages, tool definitions, response schema, and
model parameters. Role-specific minimization must explicitly allow only the
fields required for replay. Provider headers, SDK objects, transport state, and
unneeded metadata should be omitted before sanitization.

The general artifact replay verifier now maps
`LLMRequestArtifactLinked.request_artifact_id` exclusively to `llm_request`.
Missing, late, dangling, or wrong-role links fail like every other semantic
artifact reference.

## Offline recorded-provider boundary

`build_recorded_provider_replay(events, artifact_store=...)` first runs the full
artifact journal verifier, then validates the logical-call/provider-attempt
lifecycle and returns an immutable `RecordedProviderReplay`.

For each call it preserves:

- exact sanitized canonical request bytes and complete `ArtifactRef`;
- stable operation/requested-model metadata;
- ordered started/completed/failed provider attempts, token/cost/latency data,
  and bounded failure classifications;
- exact sanitized response bytes/reference for a completed call;
- explicit completed, failed, or crash-prefix/incomplete status.

`replay_response(call_id, request_bytes=...)` returns response bytes only when
the caller supplies the exact recorded sanitized request and the recorded call
completed. A mismatch, failed call, incomplete call, missing response artifact,
invalid retry, out-of-order attempt, duplicate link, or inconsistent terminal
event fails closed. The class has no provider port, callback, URL, or fallback,
so replay cannot silently become a live model request.

Response ordering is causal rather than based only on content-addressed
registration time. Replay requires a post-success compatible registration whose
`causation_event_id` equals the exact successful attempt-completion event. An
uncaused duplicate registration cannot launder bytes registered before the
attempt, while two legitimate calls may still produce and causally register the
same content-addressed response. This proves ordering and declared causation
within the trusted append-only event model; it cannot prove that an external
provider physically emitted particular bytes.

Only `build_recorded_provider_replay` may construct the replay and its call/
attempt snapshots. Call and attempt classes are not re-exported application
constructors, and private construction tokens reject direct fabrication. The
boundary keeps separate canonical public and internal snapshots plus integrity
fingerprints; mutating a returned/public frozen object through low-level Python
APIs cannot change replay output, and mutation of an internal snapshot is
detected before use.

## Inline event-string audit

Artifact IDs alone do not prevent callers from placing prompt/response text or
exception bodies in unrelated event string fields. M1d therefore adds a closed
inline-text policy map over the complete event vocabulary. `EventEnvelope`
validates every direct `str`/`Optional[str]` payload field before append, and
module import checks that future fields cannot appear without an explicit
policy.

| Policy | Intended values | Main constraints |
|---|---|---|
| `sha256` | content/cache/config digests | exact lowercase 64-hex |
| `media_type` | artifact media type | exact ASCII string, at most 256 characters, strict type/subtype and shared credential/content rejection |
| `policy_component` | policy ID/version | short storage-safe component; credential shapes rejected |
| `enum_value` | closed event enum constants | import-audited short ASCII constant; compound prompt/content markers rejected |
| `metadata_token` | operation, fidelity, reason code, archive/operator names | ASCII token, at most 64 characters |
| `routing_label` | provider/model identifiers | ASCII routing grammar, at most 128 characters |
| `exception_type` | exception class name only | dotted ASCII identifier |
| `safe_summary` | stop/failure/cache classification summary | trimmed single-line ASCII, at most 160 bytes/24 words; no JSON/container markers, prompt-role markers, credential patterns, URLs with credentials, or long opaque runs |

Closed string coverage also includes values that do not appear as direct `str`
annotations:

- every envelope and payload `StableId` is revalidated at the envelope boundary,
  is at most 128 characters, and rejects shared credential/content signatures;
- deterministic test/replay namespaces are at most 48 characters and use the
  same non-content policy before any ID is produced;
- enum values are a closed, import-audited set of short ASCII constants;
- `Decimal` values use bounded finite fixed-point text (at most 64 coefficient
  digits, absolute exponent 64, and 128 encoded characters); and
- the only payload container is the immutable objective vector, whose nested
  names use the metadata-token policy and whose numeric values must be finite.

Envelope event type, IDs, schema/sequence/offset integers, and timestamp use
exact runtime types; timestamps must use the canonical `timezone.utc` instance.
This prevents subclass dispatch and mutable timezone implementations from
changing serialized values after validation.

The import-time value-schema audit walks every registered payload annotation,
including optional, enum, ID, Decimal, and tuple leaves. A new container,
Decimal field, nested string path, unsupported annotation, or unsafe enum/event
constant fails import until it receives an explicit durable policy. Invalid
IDs, enum values, Decimal text, payload/envelope keys, and event types are
reported with value-free codec errors so tampered JSON cannot be reflected into
operational logs. JSON syntax failures likewise discard the parser exception
context, which may retain the original document.

Serialization and both event-store adapters independently reconstruct a
canonical event snapshot before persistence. The in-memory adapter stores that
detached snapshot and returns new snapshots on every read. Consequently,
low-level mutation of a caller-owned frozen envelope/payload before append is
rejected, while mutation after append or after a read cannot rewrite history.
JSONL UTF-8/codec corruption errors are also value-free and context-free.

Objective names are separately constrained as metadata tokens. Full exception
text, stack traces, stdout/stderr, provider bodies, prompts, responses, headers,
candidate configurations, and tool payloads must use a role-specific sanitized
artifact instead of an inline field. Rejection happens while constructing the
`EventEnvelope`, before `EventStore.append`, and generic errors do not retain the
rejected value in their string, cause, or context.

This inline classifier is deliberately conservative and bounded, not a DLP
claim. A short arbitrary secret can resemble a legitimate summary. Deployment
composition must still inject known literals into artifact sanitization, avoid
deriving summaries from raw exceptions, and perform the planned inline-message
security audit before active provider wiring.

This is validation hardening within event schema version 1, not a payload-shape
migration: no serialized field or existing event payload changed. Historical
records using credential/content-bearing or overlong IDs, noncanonical Decimal
strings, or values that already violated the documented metadata boundary now
fail closed on decode. A future relaxation or representational change still
requires the held schema-migration policy.

## Verification coverage

Tests exercise canonical request-link event JSON, request/response role
compatibility, exact offline replay, retry reconstruction, failed/incomplete
calls, missing/late links, missing responses, wrong retry state, immutable
snapshots, absence of provider/store writes, and raw-secret absence from request
artifacts and errors. Replay rereads defend against store changes after initial
journal verification and discard secret-bearing adapter exception context.
Adversarial inline tests cover prompt text, response JSON, Bearer/cookie/model
credentials, opaque runs, invalid UTF-8, forged typed IDs, overlong IDs and
namespaces, envelope/payload ID decode, enum/Decimal/nested-container codec
inputs, hostile string/datetime/ID/Decimal/container subclasses, stateful
timezones, media-type prompt parameters/size, pre/post-append mutation,
factory-only replay construction, internal/public replay mutation, causal
response registration, identical response deduplication, and
failure-before-append ordering. The complete value-schema audit has a drift
regression.

## Residual HOLD items

- HOLD all active-loop/provider integration. M1d proves an offline boundary but
  does not yet establish the composition root, exact request schema per harness,
  or enforce causal request/response registration in live execution.
- HOLD production secret-safety claims pending adversarial corpus testing and an
  operational security review. Inline classification cannot recognize every
  short or novel credential.
- HOLD provider-error normalization. Live adapters must map raw exceptions to
  stable failure codes/summaries and journal separately sanitized diagnostics.
- HOLD tool-call streaming, partial responses, multi-modal/binary payloads, and
  non-JSON request or response representations.
- HOLD automatic resume/replay invocation, unreadable append recovery, orphan
  collection, authenticity/signatures, encryption/access control, and schema
  migration policy.

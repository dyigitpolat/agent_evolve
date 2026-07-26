# M1c sanitized artifact registration journal

Status: provider-free vertical slice; deliberately not connected to the active
evolution loop, prompt construction, evaluators, or model providers.

## Contract

M1c adds a journal boundary between potentially sensitive in-memory values and
the M1b content-addressed artifact store. Its write order is fixed:

1. minimize the value through an explicit role policy;
2. sanitize/redact a detached JSON representation;
3. encode canonical strict UTF-8 JSON;
4. reject the representation if it exceeds the configured byte limit;
5. durably `put_bytes` into the artifact store;
6. compare the returned reference, `stat`, re-read bytes, and a freshly computed
   full `ArtifactRef` with the pre-write expectation;
7. append a frozen `ArtifactRegistered` event through `EventRecorder`.

No artifact-store write API is reachable before step 5, and the only bytes
passed to it are the sanitized canonical encoding. The service wraps failures
from caller-supplied minimizers/sanitizers without chaining their exceptions, so
an unsafe extension cannot leak raw values through the normal error string or
exception cause/context. Storage and event failures propagate after the raw
value is no longer in their input domain.

M1c is intentionally JSON-only (`application/json`). Supporting text, binary
traces, or media requires a new typed preparation policy rather than bypassing
this path.

## Registration event

`ArtifactRegistered` carries the complete expected M1b reference:

- typed `ArtifactId`;
- plain SHA-256 of the exact payload bytes;
- exact byte length;
- exact media type;
- constrained `ArtifactRole`;
- storage-safe minimization policy ID/version and a SHA-256 of its canonical
  non-secret configuration;
- storage-safe sanitization policy ID and version.

The role vocabulary is closed for M1c:

| Role | Compatible semantic event fields |
|---|---|
| `run_manifest` | `RunStarted.manifest_artifact_id` |
| `candidate_configuration` | `CandidateProposed.configuration_artifact_id` |
| `llm_response` | `LLMCallCompleted.response_artifact_id` |
| `diagnostics` | diagnostics fields on run, LLM, provider-attempt, and evaluation failure events |

M1d additively extends this vocabulary with `llm_request`; see
`recorded_provider_replay_m1d.md`. No M1c event schema was rewritten.
For recorded-provider replay, an `llm_response` registration must set its
existing envelope `causation_event_id` to the exact successful
`ProviderAttemptCompleted` event; sequence order alone is insufficient.

Role and media type are distinct. Two JSON objects are not substitutable merely
because their byte representation has the same format.

## Default safety policies

`TopLevelAllowlistMinimizer` requires an allowlist for every role a caller uses.
There is no implicit pass-through role. A custom minimizer can implement nested
schemas or domain-specific summaries behind the small `ArtifactMinimizer` port.
The port also supplies policy ID/version and a canonical configuration digest;
the default digest covers the complete role-to-field allowlist without putting
field names into the event. A custom implementation owns the truthfulness and
stability of its declared digest.

`StrictJsonSanitizer` is deterministic and performs all work in memory. It does
not inspect `.env`, environment variables, files, or network services. It:

- recursively copies supported JSON values;
- replaces values under common secret-bearing field names, including cloud
  access/secret/session keys, webhook secrets, and signing keys;
- replaces occurrences of exact secret strings injected by the composition
  root, longest first (literals shorter than eight characters or overlapping
  the redaction marker are rejected as unsafe policy inputs);
- redacts common Bearer/Basic headers, credential assignments, cookies, known
  token prefixes, AWS access-key IDs, and JWT-shaped strings;
- enforces strict UTF-8, finite numbers, bounded nesting/node count, per-string
  bytes, and aggregate bytes across retained strings/keys before regex work, and
  rejects cycles or non-JSON values;
- rejects private-key markers, credential-bearing URLs, exact secrets in object
  keys, credential-shaped object keys, and any configured literal that remains
  after redaction.

Errors intentionally report neither values, field paths, matches, nor match
counts. Exact secret values are not placed in object representations or event
metadata.

Policy metadata is a trusted composition input. The domain rejects common
credential-shaped policy IDs/versions as a second line of defense, but callers
must still never derive provenance labels from secret values; arbitrary secrets
cannot be distinguished perfectly from legitimate opaque identifiers.

This is a bounded redaction policy, not a claim of perfect secret detection or a
replacement for a DLP/security review. High-entropy strings without a known
shape can still be sensitive. Deployment composition must inject every known
credential literal and choose narrow role allowlists; suspicious residual forms
fail closed instead of being declared safe.

## Crash and observer behavior

| Failure boundary | Durable artifact | Registration event | Consequence |
|---|---:|---:|---|
| minimization, sanitization, encoding, or size gate | no | no | safe rejection |
| store reports failure | not promised | no | an atomic rename may have produced an invisible store-only orphan |
| injected crash hook immediately after successful put | yes | no | store-only orphan |
| post-write verification fails | possibly | no | store-only artifact is not trusted |
| event append fails before commit and absence is readable | yes | no | store-only orphan |
| event append raises after the exact event is readable | yes | yes | recorder advances its sequence, skips sinks, and raises `EventAppendObservedError`; durability remains adapter-reported as failed |
| event append outcome cannot be read/reconciled | yes | unknown | `EventAppendReconciliationError`; reconstruct/recover the recorder before continuing |
| event sink fails | yes | yes | registration remains durable; observer failure propagates |
| later semantic event is never appended | yes | yes | valid unreferenced registration/orphan |

The post-put hook receives only `ArtifactRef`, never payload bytes. It exists to
make this crash boundary executable in tests; it is not a business workflow
hook.

## Replay verifier

`verify_artifact_journal` is read-only and deterministic. For one contiguous run
stream it verifies each registration against both `stat` and re-read bytes,
recomputes the typed content address, strict-decodes the JSON, requires its
canonical re-encoding to equal the stored bytes, and then walks semantic
references in sequence order. A reference is valid only when a compatible
registration precedes it. Invalid UTF-8/JSON, noncanonical JSON, dangling, late,
wrong-role, media-tampered, metadata-tampered, and store-mismatched references
fail.

The role map is explicit and fail-closed. The complete registered event
vocabulary is validated when the replay module loads, not merely when a payload
type happens to occur in one run. A new `ArtifactId` field without a role rule,
or a non-scalar artifact-reference collection without a dedicated replay design,
raises `ArtifactReferenceSchemaError`. Unreferenced registration events are
accepted. Store-only orphans cannot be enumerated through the minimal M1b store
port and are intentionally invisible to replay.

## Example composition

```python
minimizer = TopLevelAllowlistMinimizer({
    ArtifactRole.DIAGNOSTICS: {"message", "exception_type", "context"},
})
sanitizer = StrictJsonSanitizer(
    exact_secret_values=(runtime_openrouter_key,),
    policy_id="strict-json-redaction",
    policy_version="1",
)
journal = ArtifactJournal(
    artifact_store=artifact_store,
    event_recorder=event_recorder,
    minimizer=minimizer,
    sanitizer=sanitizer,
    max_size_bytes=256_000,
)
result = journal.register_json(value, role=ArtifactRole.DIAGNOSTICS)
```

The composition root must obtain `runtime_openrouter_key`; the sanitizer itself
must never load it from process or filesystem state.

## Verification coverage

The M1c tests cover deterministic event JSON, recursive redaction, literal and
pattern credentials, strict UTF-8, residual high-risk rejection, explicit
allowlisting, write ordering, size-before-put, store/append/sink failures, the
post-put crash hook, absence of raw secrets from stored bytes/events/errors,
full-reference and media tampering, dangling/late/wrong-role references,
unreferenced registrations, and deterministic replay.

## Residual HOLD items

- HOLD active-loop integration until M1c and the surrounding event vocabulary
  have completed independent review. Legacy paths can still write diagnostics or
  responses without this journal because this slice intentionally does not wire
  them.
- HOLD active-loop prompt/request wiring. M1d adds the offline role, link event,
  and replay verifier, but legacy/provider execution paths remain intentionally
  disconnected.
- HOLD production claims about secret safety. The default policy is bounded and
  requires deployment-specific allowlists, injected literal secrets, adversarial
  corpus tests, and an operational security review.
- HOLD non-JSON artifacts until each representation has a typed minimization and
  sanitization contract.
- HOLD crash-orphan garbage collection. The current artifact-store port has no
  enumeration/delete API; adding one must preserve immutable audit evidence.
- HOLD authenticity, encryption, access control, signatures, hash-chained event
  manifests, distributed transactions, and remote/object-store adapters. M1c
  proves integrity and ordering only within the injected M1b/M1 event contracts.
- HOLD automatic replay on resume/startup and schema migration policy. M1c
  exposes the verifier but does not decide when a future runtime must invoke it.

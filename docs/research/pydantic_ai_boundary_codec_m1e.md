# Pydantic AI boundary codec (M1e offline slice)

Status: offline codec and framework-retry slice only. Provider journaling,
recorded-model replay, the explicit OpenRouter client, and live composition are
held.

## Closed production route

The scored-route request projector accepts only the exact Pydantic AI 1.107.1
`OpenAIChatModel` adapter backed by the exact `OpenRouterProvider` type and an
exact `OpenRouterModelProfile`. The request records:

- model adapter, provider, and profile qualified type identities;
- every field of `OpenRouterModelProfile`, including its OpenAI-compatible and
  OpenRouter cache-capability fields, not only its base profile fields;
- model name/system, model defaults, per-call settings, and the complete narrow
  `ModelRequestParameters` projection;
- exact Pydantic AI, Pydantic, and Pydantic Core versions; and
- a format-v3 output-type contract, bound to the exact runtime output object,
  plus its canonical bytes and SHA-256 digest.

Timestamp, run-ID, conversation-ID, and user-part timestamp omission is scoped
only to this exact OpenRouter/OpenAI-chat route. Inspection of the pinned chat
adapter establishes that its one-message text mapping does not consume those
fields. The exact `FunctionModel` route used by offline sentinel tests retains
all four fields and is never eligible for scored/live execution.

The boundary is before provider `prepare_request`. A structured Pydantic output
therefore remains `auto`; its complete OpenAI profile records the default
resolution to `tool`. This is a Pydantic request-boundary identity, not an HTTP
wire-byte claim.

## Output contract

Before `Agent.run_sync`, the harness factory builds a `BoundOutputContract`
containing the exact runtime `output_type`, canonical Pydantic JSON schema, a
closed recursive type graph, and the exact pre-prepare output object/tool shape
compiled by the pinned Pydantic AI output schema implementation. Output-contract
format version 3 makes the pinned 1.107.1 runtime binding an explicit
compatibility break. Request and response envelopes likewise use boundary
format version 2 because Pydantic AI renamed built-in tools to native tools and
expanded the exact profile/tool-definition dataclasses.
Standalone or recomputed bytes are not a request-codec input. Validation,
request projection, parameter projection, and request encoding each rebuild the
canonical payload from the bound runtime object before use. The harness also
checks the Agent-compiled JSON schema before execution, and observed output
objects/tools are compared with the recorded request shape as canonical bytes.

Recorded replay must call `bind_recorded_output_type_contract` with an
independently selected expected runtime output type. The binder rebuilds that
type and accepts the recording only when both bytes and digest are exact. It
returns a fresh bound object; it never turns serialized bytes alone into a
consumable contract.

The first slice accepts only `str`, `list[str]`, or a Pydantic `BaseModel`
subclass at the top level. Model fields form a deliberately closed grammar:
exact scalar leaves (`str`, `int`, `float`, `bool`, `Decimal`, and `None`),
Enums with strict JSON member values, resolved nested Pydantic models,
`list[T]`, `dict[str, T]`, fixed tuples, unions/optionals, `Literal`, and
`Annotated` with reviewed constraint metadata. Recursive model references are
recorded explicitly and traversal has depth and node budgets. `Any`, unresolved
forward references, TypedDicts, dataclass/arbitrary classes, variable tuples,
unsupported generic origins, opaque metadata, and field default factories fail
closed. Exact empty `list`/`dict` factories are the only exception and their
identity is recorded in the graph. Other top-level union/list specifications, image output,
native/prompted modes, and arbitrary function/builtin tools are also rejected.

Constraint metadata is accepted only when its runtime type is an exact
package-owned dataclass captured from pinned `annotated-types` or
`pydantic.types`, or the exact pinned Pydantic general-metadata carrier. Every
metadata attribute must itself be strict JSON and is recorded by value. The
canonical JSON Schema independently binds the emitted constraint semantics;
an arbitrary qualified type name or `repr` is never treated as sufficient.
Scalar, generic-origin, built-in-container, trusted-metadata, and recursive
model checks use identity comparisons or ID-keyed traversal state. User-defined
metaclass equality and hashing therefore cannot impersonate a supported scalar,
`Union`, metadata carrier, built-in container, or active recursive model.

`DeferredToolRequests` and `DeferredToolResults` are forbidden as exact types,
subclasses, or instances at every traversed position, including nested models,
containers, unions, literals, annotated bases/metadata, defaults, and trusted
metadata attributes. This check occurs before `Agent` construction.

## Strictness and replay safety

Message and parameter dataclasses have closed field-set checks. Runtime
containers, booleans, literal discriminators, output tools, and JSON values use
exact types; truthiness and Pydantic coercion are not used as exclusion gates.
The request projection is canonicalized and decoded once before return so no
caller-owned nested list/dict is retained.

Response envelopes require canonical bytes and exact envelope field types. A
validated `ModelResponse` is dumped into a normalized envelope and compared as
canonical bytes, preventing Python's `True == 1 == 1.0` equality from admitting
coercion. Unknown fields, aliases, noncanonical JSON, and dependency/version
drift fail with a bounded value-free error.

Public codec entry points use a two-frame sentinel pattern: a private attempt
frame catches and discards all lower-level failures, then the public frame raises
the fixed boundary error after no exception handler is active. Adversarial tests
assert both `__cause__` and `__context__` are `None` and that injected
credential-like exception text is absent from `str` and `repr`.

The public harness execution boundary applies the same separation to
`Agent.run_sync`. A private helper consumes framework failures, including the
`ExceptionGroup` that Pydantic AI can retain as exception context, and the
harness raises a fixed `HarnessOutputError` only after that helper returns.

## Exact scope of the identity claim

The format-v3 output-contract bytes and format-v2 boundary bytes identify the
pinned provider-bound Pydantic request schema
and pre-prepare output shape. They do **not** durably identify Python validator,
serializer, or Pydantic Core hook code. A custom JSON-schema hook is allowed
when its actual emitted provider schema and request shape are recorded and
revalidated. Conversely, two same-qualified-name classes with the same fields
and schema can have opposite validators or core hooks while producing identical
canonical bytes. Their in-memory bound contracts remain distinct because their
runtime type objects are not identical, but a cross-process recording cannot
prove which code produced those objects.

Any scored or replayed workflow that uses custom validators, serializers, or
core hooks therefore needs separate code/manifest provenance and must supply
the expected runtime class from that verified environment. M1e makes no broader
validator-behavior or source-code identity claim.

## Retry and live HOLD

The legacy harness now constructs `Agent(retries=0)`, which freezes both tool
and output retries at zero on the pinned Pydantic AI version. The
application loop remains the only configured framework-level retry owner.
This does not prove the provider SDK has zero retries: live composition still
requires an explicit `AsyncOpenAI(max_retries=0)` client, durable request/event
ordering, response sanitization, a provider-incapable recorded model, and an
honest typed-validation-failure lifecycle. No live/provider use is authorized
by this slice.

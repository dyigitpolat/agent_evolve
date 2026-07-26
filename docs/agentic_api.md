# Agentic benchmark API

Third-party benchmarks should import domain contracts and composition helpers
from `agent_evolve.agentic`. Provider integrations remain separate: the public
builder accepts any `AgenticGenerator`, planner, ID factory, and memory bank.

```python
from agent_evolve.agentic import (
    AgenticBenchmark,
    OptimizerBudget,
    RewardPolicyBinding,
    compose_agentic_optimizer,
)

benchmark = AgenticBenchmark(
    problem=problem,                         # Problem + Pydantic candidate_model
    detailed_evaluator=evidence_adapter,     # evaluate_evidence(config)
    phenotype_identity=phenotype_policy,     # physical-evaluation cache identity
    outcome_relation=relation_binding,       # explicit order over evidence
    reward=RewardPolicyBinding(reward, reward_definition_sha256),
    finite_variation_catalogs=(shape, trim, union),
)

# A planner binds a named, benchmark-owned catalog to its exact parent.
contract = benchmark.bind_finite_variation("shape_catalog", parent.configuration)

composition = compose_agentic_optimizer(
    benchmark,
    generator=generator,
    planner=planner,
    budget=OptimizerBudget(32, 12, 4),
    seed=7,
)
result = await composition.optimizer.run(seed_configurations)
```

Detailed mode fails closed unless the evaluator and outcome relation are both
present. The builder gives the engine and archive the same relation object, and
its planner guard authenticates each wave's reward definition and every finite
catalog contract. Catalog IDs must be unique within a benchmark; catalog,
evaluator, phenotype-policy, and objective identity drift is rejected before a
run starts.

See
[`examples/development/public_agentic_api_smoke.py`](../examples/development/public_agentic_api_smoke.py)
for a complete provider-free construction example.

## Deferred runtime factories and post-G3 curation

Stateful policies may be constructed only after the composition root owns the
authoritative collaborators. `GenerationPlannerFactory.build` receives the
benchmark, engine, ID factory, and memory. The subsequently invoked
`GenerationFeedbackInterceptorFactory.build` receives those same objects plus
the exact runtime `planner`; factories should verify object identity rather than
reconstructing or passing a planner through mutable side state.

The generic G3 postseal policy injects benchmark semantics through only an
actionable reflection contract and a declarative source-slot scope:

```python
from agent_evolve.agentic import (
    G3CurationSourceScope,
    G3PostsealCurationFactory,
    G3PostsealCurationSpec,
    ReflectionInsightContract,
)

curation_spec = G3PostsealCurationSpec(
    insight_contract=ReflectionInsightContract(
        required_metric_ids=("objective:drag", "violation:lift"),
        allowed_option_families=("trim_only",),
        allowed_option_ids=("trim.presealed_action",),
    ),
    # Use only evidence that can honestly carry the exact action attribution.
    # All G1--G3 generation receipt hashes are bound even for this projection.
    source_scope=G3CurationSourceScope(
        policy_id="direct_adaptive_only",
        policy_version=1,
        policy_definition_sha256=scope_definition_sha256,
        slot_ids=("g2_adaptive",),
    ),
)

composition = compose_agentic_optimizer(
    benchmark,
    generator=generator,
    planner_factory=planner_factory,
    feedback_interceptor_factory=G3PostsealCurationFactory(
        spec=curation_spec,
    ),
    budget=G3_SCREEN_BUDGET,
    seed=7,
)
```

The sixth call uses `reflect_with_receipt`. Its engine-issued typed receipt binds
the actual prompt/request, source receipts and projected outcome hashes,
contract, predecessor, provider telemetry, and any zero-prior quarantined
revision. Final G3 validation requires the exact curation spec, pre-provider
authority, and curation receipt; it also fetches the engine-stored call receipt
and verifies current memory lineage. A completed zero-card response is recorded
as `completed_abstention`, not as revision efficacy. Provider/cardinality
failure is charged as the sixth logical call and isolated as `incomplete/failed`
without changing sealed G1--G3 endpoints.

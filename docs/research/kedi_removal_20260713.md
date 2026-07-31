# Kedi removal (2026-07-13)

The obsolete in-repository Kedi path has been removed from the active rewrite:

- removed the nested gitlink and `.gitmodules` declaration;
- removed the optional dependency, package data, bootstrap registration, adapter,
  DSL program, Kedi-only helpers/tests, and Kedi CLI guidance;
- reduced `Directives` to its runtime-independent `compose_*` prompt contract;
- retained the generic `Harness`, `HarnessRegistry`, integration entry points,
  and `optimize(harness=...)` selection seam; and
- replaced Kedi equivalence checks with a deterministic, provider-free
  `PydanticAIHarness` typed-output path versus `FakeHarness`.

Validation:

```text
PYTHONSAFEPATH=1 ../env/bin/python -m pytest tests/ -q
64 passed at validation time
```

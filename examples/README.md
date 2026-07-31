# Examples

Two worked examples. Both run with **no credentials, no network and no cost**.

| Example | Read it for |
| --- | --- |
| [`build_flags/`](build_flags/) | The regime this tool suits: categorical choices, awkward constraints, an expensive evaluator. Start here. |
| [`knapsack/`](knapsack/) | The five obligations in the shortest possible problem. `--compare` runs the model against the free baseline. |

```bash
python examples/build_flags/run.py
python examples/knapsack/run.py --compare
```

Nothing else in this directory is an example.

`development/` and `benchmarks/` are the research campaign scripts -- about a
hundred and eighty of them, one 271 KB. **Do not read them to learn the API.**
They are here, and not in a directory named for what they are, for a reason
worth stating: sixty-six files refer to them by *filesystem path* rather than
by import, ten of those in the paper's own repository. Renaming the directory
would break the property that makes a release claim checkable -- that the
shipped tree and the measured tree are the same tree. Tidiness is not worth
that, so they keep the misleading name and this note exists instead.

They are excluded from the distribution: `pip install agent_evolve` ships
`src/` only, so nobody who installs the package ever sees them.

To learn the API, read the two examples above, then [`../docs/`](../docs/).

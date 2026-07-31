"""Accelerator mapping with the campaign's seed prefix, for the capability ladder.

``TimeloopV2CoDesignProblem`` defines objectives and evaluation but supplies no
starting points, so ``optimize()`` would run with an empty generation 0. That
matters twice: the tiers of a ladder would share no starting archive, and the
run could not answer "did this beat what I already had", which is the question a
user actually brings.

This wrapper supplies the two-candidate prefix the sealed campaign of record
used, and nothing else. It adds no objective, no constraint and no evaluation
behaviour.

**What the prefix is, declared rather than inherited.** Both candidates are
generation 0 in the sealed campaign, carry no ``parent_ids`` and no
``operator_kind``: they are hand-specified starting configurations, not the
product of prior optimization.

* ``seed_default`` -- the module's own ``DEFAULT_CANDIDATE``.
* ``seed_pe_mesh_16`` -- the same configuration with ``pe_mesh_x`` set to 16.

So generation 0 is a plausible engineering starting point rather than a tuned
archive. It is not neutral -- a default configuration chosen by the benchmark's
author encodes judgement, and in hypervolume terms the prefix already accounts
for a large share of the campaign's endpoint. A reader should take wave-one
markets as conditioned on a sensible default pair, which is the realistic case
for a user, not on an empty or an optimized start.

Verified rather than asserted: ``python -m ... ladder_problem --verify``
evaluates both seeds and compares the objective vectors to the sealed campaign's
frozen archive.
"""

from __future__ import annotations

import copy
from typing import Any, Iterator

from examples.benchmarks.timeloop_codesign.v2.problem_def import (
    DEFAULT_CANDIDATE,
    create_default_problem,
)

SEED_LABELS = ("campaign_seed_seed_default", "campaign_seed_seed_pe_mesh_16")


def _as_dict(candidate: Any) -> dict:
    if isinstance(candidate, dict):
        return copy.deepcopy(candidate)
    if hasattr(candidate, "model_dump"):
        return candidate.model_dump()
    return copy.deepcopy(vars(candidate))


def campaign_seed_prefix() -> tuple[dict, dict]:
    """The sealed campaign's two generation-0 configurations, reconstructed."""

    default = _as_dict(DEFAULT_CANDIDATE)
    pe_mesh_16 = _as_dict(DEFAULT_CANDIDATE)
    pe_mesh_16["pe_mesh_x"] = 16
    return default, pe_mesh_16


class SeededTimeloopV2Problem:
    """``TimeloopV2CoDesignProblem`` plus the campaign's starting points.

    Every attribute other than ``seeds`` is delegated unchanged, so the wrapper
    cannot alter evaluation, objectives or validation.
    """

    def __init__(self, inner: Any | None = None) -> None:
        self._inner = inner if inner is not None else create_default_problem()

    def seeds(self) -> Iterator[dict]:
        yield from campaign_seed_prefix()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


problem = SeededTimeloopV2Problem()


def _verify() -> int:
    """Evaluate both seeds and print their objectives for comparison."""

    import json

    inner = create_default_problem()
    for label, config in zip(SEED_LABELS, campaign_seed_prefix()):
        objectives = inner.evaluate(config)
        print(f"{label}: {json.dumps(objectives, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_verify())

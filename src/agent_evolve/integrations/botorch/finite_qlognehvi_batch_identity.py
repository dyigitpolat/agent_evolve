"""Dependency-free identity of the direct qLogNEHVI batch scorer."""

from __future__ import annotations

import hashlib


POLICY_ID = "botorch_finite_qlognehvi_batch_score"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:botorch-finite-qlognehvi-batch-score:v1;"
    b"model=independent-output-singletask-gp-standardized;"
    b"noise=declared-1e-6;acquisition=qlognehvi;"
    b"reference=normalized-zero-vector;sampler=sobol-common-seed;"
    b"input=sealed-equal-cardinality-slates;selection=false;"
    b"all-slates-scored-under-one-fit-and-sampler=true;cpu-double=true"
).hexdigest()


__all__ = ["POLICY_DEFINITION_SHA256", "POLICY_ID", "POLICY_VERSION"]

"""Dependency-free identity for the optional BoTorch qLogNEHVI expert."""

from __future__ import annotations

import hashlib


POLICY_ID = "botorch_finite_qlognehvi"
POLICY_VERSION = 1
POLICY_DEFINITION_SHA256 = hashlib.sha256(
    b"agent-evolve:botorch-finite-qlognehvi:v1;"
    b"model=independent-single-task-gp-known-noise-1e-6;"
    b"outcome-transform=standardize;input-frame=adapter-normalized-unit-cube;"
    b"objective-frame=fixed-affine-reference-zero-ideal-one;"
    b"acquisition=qlog-noisy-expected-hypervolume-improvement;"
    b"optimizer=exact-finite-discrete-unique;"
    b"real-observations=prior-cutoff-only;"
    b"configuration-materialization=outside-policy;workload-branches=false"
).hexdigest()


__all__ = ["POLICY_DEFINITION_SHA256", "POLICY_ID", "POLICY_VERSION"]

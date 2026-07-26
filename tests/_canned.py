"""Shared test typing for canned candidate factories."""

from __future__ import annotations

from typing import Any, Callable, Dict

CandidateFactory = Callable[[int], Dict[str, Any]]

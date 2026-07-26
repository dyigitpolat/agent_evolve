"""Wall/monotonic clock port."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def utc_now(self) -> datetime: ...
    def monotonic_ns(self) -> int: ...


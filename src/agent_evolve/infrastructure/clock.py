"""System and deterministic clock implementations."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone


class SystemClock:
    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


class FakeClock:
    """Manually advanced clock; reads have no hidden side effects."""

    def __init__(
        self,
        start_utc: datetime | None = None,
        *,
        start_monotonic_ns: int = 0,
    ) -> None:
        start_utc = start_utc or datetime(2000, 1, 1, tzinfo=timezone.utc)
        if start_utc.utcoffset() != timedelta(0):
            raise ValueError("start_utc must be timezone-aware UTC")
        if (
            isinstance(start_monotonic_ns, bool)
            or not isinstance(start_monotonic_ns, int)
            or start_monotonic_ns < 0
        ):
            raise ValueError("start_monotonic_ns must be a non-negative integer")
        self._utc = start_utc
        self._monotonic_ns = start_monotonic_ns
        self._lock = threading.Lock()

    def utc_now(self) -> datetime:
        with self._lock:
            return self._utc

    def monotonic_ns(self) -> int:
        with self._lock:
            return self._monotonic_ns

    def advance_ns(self, nanoseconds: int) -> None:
        if isinstance(nanoseconds, bool) or not isinstance(nanoseconds, int) or nanoseconds < 0:
            raise ValueError("nanoseconds must be a non-negative integer")
        with self._lock:
            self._monotonic_ns += nanoseconds
            self._utc += timedelta(microseconds=nanoseconds / 1_000)

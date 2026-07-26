"""Import-stable phase-boundary contracts for the BOiLS action shadow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProposalClosureReceipt:
    """Capability issued from a durable phase-close record after queue closure.

    This type lives outside the executable composition root so a direct script
    launch and the delayed scorer share one Python class identity.
    """

    queue_closed: bool
    terminal_logical_calls: int
    terminal_response_hashes: tuple[str, ...]
    closure_event_sha256: str

    def __post_init__(self) -> None:
        if self.queue_closed is not True:
            raise ValueError("proposal closure receipt requires a closed queue")
        if self.terminal_logical_calls != 12:
            raise ValueError("proposal closure receipt requires twelve terminal calls")
        if (
            len(self.terminal_response_hashes) != 12
            or any(
                type(value) is not str or len(value) != 64
                for value in self.terminal_response_hashes
            )
        ):
            raise ValueError("proposal closure receipt has invalid response hashes")
        if (
            type(self.closure_event_sha256) is not str
            or len(self.closure_event_sha256) != 64
        ):
            raise ValueError("proposal closure receipt has an invalid event hash")


__all__ = ["ProposalClosureReceipt"]

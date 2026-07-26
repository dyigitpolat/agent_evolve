"""Framework-free identities for versioned research insights."""

from __future__ import annotations

from dataclasses import dataclass

from agent_evolve.domain.ids import InsightId


@dataclass(frozen=True, slots=True, order=True)
class InsightRef:
    """An immutable reference to one exact version of an insight.

    Revisions retain the logical :class:`InsightId` and increment ``version``.
    Selection and credit always bind the exact version so later prose edits
    cannot inherit evidence silently.
    """

    insight_id: InsightId
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.insight_id, InsightId):
            raise TypeError("insight_id must be an InsightId")
        if type(self.version) is not int or self.version <= 0:
            raise ValueError("insight version must be a positive integer")


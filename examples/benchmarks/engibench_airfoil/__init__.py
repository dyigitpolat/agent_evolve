"""EngiBench Airfoil external smooth-panel benchmark."""

from .problem_def import (
    AirfoilPanelCandidate,
    AirfoilPanelProblem,
    AirfoilPanelSettings,
    create_default_problem,
)

__all__ = [
    "AirfoilPanelCandidate",
    "AirfoilPanelProblem",
    "AirfoilPanelSettings",
    "create_default_problem",
]

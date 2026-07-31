"""Optional direct and dependency-isolated BoTorch acquisition integrations."""

from agent_evolve.integrations.botorch.subprocess_qlognehvi import (
    FiniteAcquisitionSubprocessError,
    IsolatedBotorchQLogNehviFiniteAcquisition,
    build_isolated_botorch_qlognehvi,
)


def __getattr__(name: str) -> object:
    # Keep torch and BoTorch optional in the main agent runtime.  The direct
    # policy is imported only by users (and the isolated worker) that request it.
    if name != "BotorchQLogNehviFiniteAcquisition":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from agent_evolve.integrations.botorch.finite_qlognehvi import (
        BotorchQLogNehviFiniteAcquisition,
    )

    globals()[name] = BotorchQLogNehviFiniteAcquisition
    return BotorchQLogNehviFiniteAcquisition


__all__ = [
    "BotorchQLogNehviFiniteAcquisition",
    "FiniteAcquisitionSubprocessError",
    "IsolatedBotorchQLogNehviFiniteAcquisition",
    "build_isolated_botorch_qlognehvi",
]

"""Numerically stable qNEHVI over a sealed finite candidate pool."""

from __future__ import annotations

import math
from dataclasses import dataclass

import botorch
import gpytorch
import torch
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from agent_evolve.ports.finite_acquisition import (
    FiniteAcquisitionDecision,
    FiniteAcquisitionRequest,
    FiniteAcquisitionSelection,
)
from agent_evolve.integrations.botorch.finite_qlognehvi_identity import (
    POLICY_DEFINITION_SHA256,
    POLICY_ID,
    POLICY_VERSION,
)


def _training_tensors(
    request: FiniteAcquisitionRequest,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.tensor(
        [value.features for value in request.observations],
        dtype=dtype,
        device=torch.device("cpu"),
    )
    axis_by_metric = {value.metric_id: value for value in request.objectives}
    train_y = torch.tensor(
        [
            [
                axis_by_metric[metric_id].maximize_value(raw_value)
                for metric_id, raw_value in observation.objectives
            ]
            for observation in request.observations
        ],
        dtype=dtype,
        device=train_x.device,
    )
    # Current benchmark evaluators are deterministic but numerically finite.
    # A small declared observation variance avoids treating floating-point
    # objective resolution as mathematical exactness.
    train_yvar = torch.full_like(train_y, 1.0e-6)
    return train_x, train_y, train_yvar


@dataclass(frozen=True, slots=True)
class BotorchQLogNehviFiniteAcquisition:
    """Fit one past-only GP and select an exact unique finite batch."""

    mc_samples: int = 128
    maximum_optimizer_batch_size: int = 2048
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.mc_samples) is not int or self.mc_samples < 16:
            raise ValueError("mc_samples must be an exact integer of at least 16")
        if (
            type(self.maximum_optimizer_batch_size) is not int
            or self.maximum_optimizer_batch_size < 1
        ):
            raise ValueError("maximum_optimizer_batch_size must be positive")
        if (
            self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("qLogNEHVI policy identity drifted")

    def select(self, request: FiniteAcquisitionRequest) -> FiniteAcquisitionDecision:
        self.__post_init__()
        if type(request) is not FiniteAcquisitionRequest:
            raise TypeError("request must be an exact FiniteAcquisitionRequest")
        request.__post_init__()
        dtype = torch.double
        train_x, train_y, train_yvar = _training_tensors(request, dtype=dtype)
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            train_Yvar=train_yvar,
            outcome_transform=Standardize(m=len(request.objectives)),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]),
            seed=request.seed,
        )
        acquisition = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=torch.zeros(
                len(request.objectives),
                dtype=dtype,
                device=train_x.device,
            ),
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
            cache_root=True,
        )
        choices = torch.tensor(
            [value.features for value in request.candidates],
            dtype=dtype,
            device=train_x.device,
        )
        selected_x, acquisition_values = optimize_acqf_discrete(
            acq_function=acquisition,
            q=request.batch_size,
            choices=choices,
            max_batch_size=self.maximum_optimizer_batch_size,
            unique=True,
            return_acq_values=True,
        )
        candidate_by_features = {
            value.features: value for value in request.candidates
        }
        selected_rows = []
        flat_values = acquisition_values.detach().cpu().reshape(-1).tolist()
        if len(flat_values) == 1 and request.batch_size > 1:
            flat_values = flat_values * request.batch_size
        if len(flat_values) != request.batch_size:
            raise RuntimeError("BoTorch returned an unexpected acquisition shape")
        for feature_tensor, acquisition_value in zip(
            selected_x.detach().cpu(),
            flat_values,
            strict=True,
        ):
            feature_key = tuple(float(value) for value in feature_tensor.tolist())
            candidate = candidate_by_features.get(feature_key)
            if candidate is None:
                raise RuntimeError("BoTorch selected a row outside the finite pool")
            value = float(acquisition_value)
            if not math.isfinite(value):
                raise RuntimeError("BoTorch returned a non-finite acquisition value")
            selected_rows.append(
                FiniteAcquisitionSelection(
                    candidate_id=candidate.candidate_id,
                    configuration_sha256=candidate.configuration_sha256,
                    acquisition_value=value,
                )
            )
        if len({value.candidate_id for value in selected_rows}) != request.batch_size:
            raise RuntimeError("BoTorch did not return a unique exact batch")
        return FiniteAcquisitionDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            selected=tuple(selected_rows),
            diagnostics=tuple(
                sorted(
                    (
                        ("botorch_version", str(botorch.__version__)),
                        ("candidate_pool_size", str(len(request.candidates))),
                        ("feature_dimension", str(train_x.shape[-1])),
                        ("gpytorch_version", str(gpytorch.__version__)),
                        ("mc_samples", str(self.mc_samples)),
                        ("objective_count", str(train_y.shape[-1])),
                        ("observation_count", str(train_x.shape[-2])),
                        ("torch_version", str(torch.__version__)),
                    )
                )
            ),
        )


__all__ = [
    "BotorchQLogNehviFiniteAcquisition",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
]

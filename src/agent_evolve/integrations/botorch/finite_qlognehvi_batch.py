"""Common-realization qLogNEHVI values for sealed finite slates."""

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
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from agent_evolve.integrations.botorch.finite_qlognehvi import _training_tensors
from agent_evolve.integrations.botorch.finite_qlognehvi_batch_identity import (
    POLICY_DEFINITION_SHA256,
    POLICY_ID,
    POLICY_VERSION,
)
from agent_evolve.ports.finite_acquisition_batch import (
    FiniteAcquisitionBatchScoreDecision,
    FiniteAcquisitionBatchScoreRequest,
    FiniteAcquisitionSlateScore,
)


@dataclass(frozen=True, slots=True)
class BotorchQLogNehviFiniteBatchScorePolicy:
    """Fit once and score every supplied slate under one acquisition object."""

    mc_samples: int = 128
    maximum_score_batch_size: int = 512
    policy_id: str = POLICY_ID
    policy_version: int = POLICY_VERSION
    definition_sha256: str = POLICY_DEFINITION_SHA256

    def __post_init__(self) -> None:
        if type(self.mc_samples) is not int or self.mc_samples < 16:
            raise ValueError("mc_samples must be an exact integer of at least 16")
        if (
            type(self.maximum_score_batch_size) is not int
            or self.maximum_score_batch_size < 1
        ):
            raise ValueError("maximum_score_batch_size must be positive")
        if (
            self.policy_id != POLICY_ID
            or self.policy_version != POLICY_VERSION
            or self.definition_sha256 != POLICY_DEFINITION_SHA256
        ):
            raise ValueError("qLogNEHVI batch-score policy identity drifted")

    def score(
        self,
        request: FiniteAcquisitionBatchScoreRequest,
    ) -> FiniteAcquisitionBatchScoreDecision:
        self.__post_init__()
        if type(request) is not FiniteAcquisitionBatchScoreRequest:
            raise TypeError("request must be exact")
        request.__post_init__()
        dtype = torch.double
        # GP fitting may use torch randomness internally.  The worker process is
        # isolated, but explicitly seeding here also makes direct invocations
        # replayable and keeps every compared slate on one fitted realization.
        torch.manual_seed(request.seed)
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
        candidate_by_id = {value.candidate_id: value for value in request.candidates}
        scores: list[FiniteAcquisitionSlateScore] = []
        for start in range(0, len(request.slates), self.maximum_score_batch_size):
            slate_chunk = request.slates[
                start : start + self.maximum_score_batch_size
            ]
            rows = torch.tensor(
                [
                    [candidate_by_id[value].features for value in slate.candidate_ids]
                    for slate in slate_chunk
                ],
                dtype=dtype,
                device=train_x.device,
            )
            with torch.no_grad():
                values = acquisition(rows).detach().cpu().reshape(-1).tolist()
            if len(values) != len(slate_chunk):
                raise RuntimeError("BoTorch returned an unexpected slate-score shape")
            for slate, raw in zip(slate_chunk, values, strict=True):
                value = float(raw)
                if not math.isfinite(value):
                    raise RuntimeError("BoTorch returned a non-finite slate score")
                scores.append(
                    FiniteAcquisitionSlateScore(
                        slate=slate,
                        log_acquisition_value=value,
                    )
                )
        return FiniteAcquisitionBatchScoreDecision(
            request_sha256=request.request_sha256,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            policy_definition_sha256=self.definition_sha256,
            scores=tuple(scores),
            diagnostics=tuple(
                sorted(
                    (
                        ("botorch_version", str(botorch.__version__)),
                        ("candidate_count", str(len(request.candidates))),
                        ("feature_dimension", str(train_x.shape[-1])),
                        ("gpytorch_version", str(gpytorch.__version__)),
                        ("mc_samples", str(self.mc_samples)),
                        ("objective_count", str(train_y.shape[-1])),
                        ("observation_count", str(train_x.shape[-2])),
                        ("slate_count", str(len(request.slates))),
                        ("slate_size", str(request.batch_size)),
                        ("torch_version", str(torch.__version__)),
                    )
                )
            ),
        )


__all__ = [
    "BotorchQLogNehviFiniteBatchScorePolicy",
    "POLICY_DEFINITION_SHA256",
    "POLICY_ID",
    "POLICY_VERSION",
]

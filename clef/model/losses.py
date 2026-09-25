from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.nn import functional as F


def _check_spectra(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("Prediction and target must have matching [batch, bins] shapes")
    if prediction.shape[0] == 0 or prediction.shape[1] != 512:
        raise ValueError("Expected a nonempty batch of 512-bin spectra")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Spectra must be floating-point tensors")
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(target).all()):
        raise ValueError("Spectra must contain only finite intensities")
    if bool((prediction < 0).any() or (target < 0).any()):
        raise ValueError("Spectral intensities must be nonnegative")


def transformed_spectrum(spectrum: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return F.normalize(torch.sqrt(spectrum + epsilon), p=2, dim=-1)


def spectrum_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-9,
) -> torch.Tensor:


    _check_spectra(prediction, target)
    transformed_prediction = transformed_spectrum(prediction, epsilon)
    transformed_target = transformed_spectrum(target, epsilon)
    return (transformed_prediction - transformed_target).square().mean()


def spectrum_loss_from_batch(
    output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]
) -> torch.Tensor:

    return spectrum_reconstruction_loss(output["spect"], batch["spect"])


def physical_sdp(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:

    _check_spectra(prediction, target)
    mass = torch.arange(1, 513, device=prediction.device, dtype=prediction.dtype)
    weight = mass.pow(3).unsqueeze(0)
    pred_weighted = weight * prediction.pow(0.6)
    target_weighted = weight * target.pow(0.6)
    numerator = (pred_weighted * target_weighted).sum(dim=1)
    denominator = torch.linalg.vector_norm(pred_weighted, dim=1) * torch.linalg.vector_norm(
        target_weighted, dim=1
    )
    return numerator / (denominator + 1e-10)

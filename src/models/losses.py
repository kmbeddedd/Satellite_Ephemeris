"""
Custom Loss Functions for Probabilistic, Physical, and Smoothness Constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Negative Log-Likelihood Loss for probabilistic forecasting.
    """
    loss = torch.log(sigma) + ((targets - mu) ** 2) / (2 * (sigma ** 2))
    return loss.mean()


def spike_bce_loss(spike_probs: torch.Tensor, spike_targets: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross Entropy loss for anomalous perturbation/spike event prediction.
    """
    return F.binary_cross_entropy(spike_probs, spike_targets)


def smoothness_loss(predictions: torch.Tensor) -> torch.Tensor:
    """
    Penalizes first-order discrete temporal acceleration in orbital trajectory predictions.
    """
    diffs = predictions[:, 1:, :] - predictions[:, :-1, :]
    return torch.mean(diffs ** 2)


def multiscale_smoothness_loss(predictions: torch.Tensor, scales: list = [1, 2, 4, 8]) -> torch.Tensor:
    """
    Enforces multi-scale temporal continuity across varying stride horizons.
    """
    total_loss = torch.tensor(0.0, device=predictions.device)
    for scale in scales:
        diffs = predictions[:, scale:, :] - predictions[:, :-scale, :]
        total_loss = total_loss + torch.mean(diffs ** 2)
    return total_loss


def frequency_consistency_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    FFT frequency domain consistency loss matching spectral power distributions.
    """
    pred_fft = torch.fft.rfft(predictions, dim=1)
    target_fft = torch.fft.rfft(targets, dim=1)
    return torch.mean((torch.abs(pred_fft) - torch.abs(target_fft)) ** 2)


def clock_drift_acceleration_loss(predictions: torch.Tensor) -> torch.Tensor:
    """
    Penalizes second-order derivative (acceleration) of satellite clock bias.
    """
    clock = predictions[:, :, 3]  # Index 3 is Error_Clock
    velocity = clock[:, 1:] - clock[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    return torch.mean(acceleration ** 2)


def composite_transformer_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    spike_probs: torch.Tensor,
    targets: torch.Tensor,
    spike_targets: torch.Tensor,
    lambda_smooth: float = 0.02,
    lambda_multi: float = 0.02,
    lambda_freq: float = 0.02,
    lambda_clock: float = 0.01,
    lambda_spike: float = 0.1,
    lambda_sigma: float = 0.01
) -> torch.Tensor:
    """
    Unified multi-task objective combining:
    - Probabilistic Gaussian NLL
    - Spike event classification
    - Temporal & multi-scale smoothness
    - Spectral FFT frequency consistency
    - Clock drift acceleration penalty
    - Dispersion regularization
    """
    nll = gaussian_nll_loss(mu, sigma, targets)
    spike = spike_bce_loss(spike_probs, spike_targets)
    smooth = smoothness_loss(mu)
    multi = multiscale_smoothness_loss(mu)
    freq = frequency_consistency_loss(mu, targets)
    clock = clock_drift_acceleration_loss(mu)
    sigma_penalty = 1.0 / (sigma.mean() + 1e-6)

    total = (
        nll
        + lambda_spike * spike
        + lambda_smooth * smooth
        + lambda_multi * multi
        + lambda_freq * freq
        + lambda_clock * clock
        + lambda_sigma * sigma_penalty
    )
    return total


def diffusion_mse_loss(predicted_noise: torch.Tensor, true_noise: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss for conditional diffusion noise prediction.
    """
    return F.mse_loss(predicted_noise, true_noise)

"""Conditional diffusion over forecast residual trajectories.

The denoiser is trained on ``target - point_forecast`` and receives that same
quantity during sampling.  Keeping the diffusion state in residual space avoids
the train/inference mismatch that occurs when a denoiser trained on residuals is
later fed ``point_forecast + residual``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pytorch_transformer import PositionalEncoding
from src.config import DIFFUSION_DEFAULTS, TARGET_COLS_4


class DiffusionSchedule:
    """
    Variance schedule for a discrete residual diffusion process.
    """
    def __init__(
        self,
        steps: int = DIFFUSION_DEFAULTS["steps"],
        beta_start: float = DIFFUSION_DEFAULTS["beta_start"],
        beta_end: float = DIFFUSION_DEFAULTS["beta_end"],
        schedule_type: str = "cosine",
        device: str = "cpu"
    ):
        self.steps = steps
        if steps < 2:
            raise ValueError("Diffusion requires at least two steps")
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        elif schedule_type == "cosine":
            # Improved-DDPM cosine schedule; unlike the prior 100-step linear
            # schedule, this reaches an approximately standard-normal terminal state.
            offset = 0.008
            grid = torch.linspace(0, steps, steps + 1, device=device) / steps
            alpha_bar = torch.cos((grid + offset) / (1.0 + offset) * math.pi / 2).square()
            alpha_bar = alpha_bar / alpha_bar[0]
            self.betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)
        else:
            raise ValueError(f"Unknown diffusion schedule: {schedule_type}")
        self.schedule_type = schedule_type
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([
            torch.ones(1, device=device, dtype=self.alpha_bars.dtype),
            self.alpha_bars[:-1],
        ])
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        ).clamp_min(1e-20)

    def forward_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        noise = torch.randn_like(x0)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_bars[t]).view(-1, 1, 1)
        noisy_x = sqrt_alpha_bar * x0 + sqrt_one_minus * noise
        return noisy_x, noise


class DiffusionTimeEmbedding(nn.Module):
    """
    Learned embedding for discrete diffusion timesteps.
    """
    def __init__(self, d_model: int = 64, steps: int = DIFFUSION_DEFAULTS["steps"]):
        super().__init__()
        self.embedding = nn.Embedding(steps, d_model)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.embedding(t)


class ConditionalDiffusionDenoiser(nn.Module):
    """
    Transformer-based denoiser conditioned on global hybrid context c and diffusion timestep t.
    """
    def __init__(
        self,
        context_dim: int = 109,
        d_model: int = 64,
        output_dim: int = len(TARGET_COLS_4),
        num_layers: int = 2,
        nhead: int = 4
    ):
        super().__init__()
        self.time_embedding = DiffusionTimeEmbedding(d_model=d_model)
        self.context_projection = nn.Linear(context_dim, d_model)
        self.input_projection = nn.Linear(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_dim)

    def forward(self, noisy_future: torch.Tensor, context: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(noisy_future)
        t_embed = self.time_embedding(t).unsqueeze(1)
        c_embed = self.context_projection(context).unsqueeze(1)
        x = x + t_embed + c_embed
        x = self.positional_encoding(x)
        x = self.transformer(x)
        predicted_noise = self.output_projection(x)
        return predicted_noise


@torch.no_grad()
def sample_diffusion_forecast(
    diffusion_model: nn.Module,
    schedule: DiffusionSchedule,
    context: torch.Tensor,
    mu: torch.Tensor,
    shape: tuple,
    noise_scale: float = 1.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generates refined multi-step forecasts via 100-step reverse DDPM process.
    """
    diffusion_model.eval()
    x = noise_scale * torch.randn(shape, device=device)

    for step in reversed(range(schedule.steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        predicted_noise = diffusion_model(x, context, t)

        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        x = (1.0 / torch.sqrt(alpha)) * (
            x - (schedule.betas[step] / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        )

        if step > 0:
            x = x + torch.sqrt(schedule.posterior_variance[step]) * torch.randn_like(x)

    return mu + x


@torch.no_grad()
def sample_ddim_forecast(
    diffusion_model: nn.Module,
    schedule: DiffusionSchedule,
    context: torch.Tensor,
    mu: torch.Tensor,
    shape: tuple,
    num_ddim_steps: int = 20,
    eta: float = 0.0,
    noise_scale: float = 1.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Fast Denoising Diffusion Implicit Models (DDIM) reverse sampler.
    Generates high-fidelity trajectories in 10-20 deterministic steps (10x faster than standard DDPM).
    eta: 0.0 for fully deterministic ODE sampling, >0.0 for stochastic Langevin noise.
    """
    diffusion_model.eval()
    B = shape[0]

    num_ddim_steps = max(2, min(int(num_ddim_steps), schedule.steps))
    times = torch.linspace(
        schedule.steps - 1, 0, num_ddim_steps, device=device
    ).round().long()
    times = torch.unique_consecutive(times)

    x_t = noise_scale * torch.randn(shape, device=device)

    for index, curr_step in enumerate(times):
        next_step = times[index + 1] if index + 1 < len(times) else None
        t_curr = torch.full((B,), int(curr_step.item()), device=device, dtype=torch.long)

        # Predict residual noise
        predicted_noise = diffusion_model(x_t, context, t_curr)

        alpha_bar_curr = schedule.alpha_bars[curr_step]
        alpha_bar_prev = (
            schedule.alpha_bars[next_step]
            if next_step is not None
            else torch.ones((), device=device, dtype=alpha_bar_curr.dtype)
        )

        # Estimated x0 (original clean residual)
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_curr) * predicted_noise) / torch.sqrt(alpha_bar_curr)

        # Direction pointing to x_t
        sigma_t = eta * torch.sqrt(torch.clamp(
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_curr)
            * (1.0 - alpha_bar_curr / alpha_bar_prev),
            min=0.0,
        ))
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=0.0)) * predicted_noise

        # DDIM update step
        noise = torch.randn_like(x_t) if eta > 0.0 and next_step is not None else 0.0
        x_t = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma_t * noise

    return mu + x_t

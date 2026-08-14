"""
Conditional Denoising Diffusion Probabilistic Model (DDPM) for Satellite Trajectories
Conditioned on Global Hybrid Context: c = [h_GRU; MHSA(H); E_PRN]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pytorch_transformer import PositionalEncoding
from src.config import DIFFUSION_DEFAULTS, TARGET_COLS_4


class DiffusionSchedule:
    """
    Linear beta schedule and alpha variance components for discrete 100-step diffusion.
    """
    def __init__(
        self,
        steps: int = DIFFUSION_DEFAULTS["steps"],
        beta_start: float = DIFFUSION_DEFAULTS["beta_start"],
        beta_end: float = DIFFUSION_DEFAULTS["beta_end"],
        device: str = "cpu"
    ):
        self.steps = steps
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

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
    noise_scale: float = 0.02,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Generates refined multi-step forecasts via 100-step reverse diffusion process.
    """
    diffusion_model.eval()
    # Start from white noise around base mean
    noise = torch.randn(shape, device=device)
    x = noise_scale * noise

    for step in reversed(range(schedule.steps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        predicted_noise = diffusion_model(mu + x, context, t)

        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        beta = schedule.betas[step]

        x = (1.0 / torch.sqrt(alpha)) * (
            x - ((1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)) * predicted_noise
        )

        if step > 0:
            extra_noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * 0.05 * extra_noise

    return mu + x

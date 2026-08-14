"""
PyTorch Deep Multi-Task Transformer Forecaster with Uncertainty & Anomaly Heads
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_4


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for time-series sequences.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class GNSSTransformerBackbone(nn.Module):
    """
    Transformer Encoder with Satellite Entity Embeddings and Multi-Head Self-Attention.
    """
    def __init__(
        self,
        num_features: int,
        num_satellites: int,
        d_model: int = 64,
        embedding_dim: int = 8,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.sat_embedding = nn.Embedding(num_satellites, embedding_dim)
        self.input_projection = nn.Linear(num_features + embedding_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        sat_embed = self.sat_embedding(sat_ids).unsqueeze(1).repeat(1, seq_len, 1)
        x = torch.cat([x, sat_embed], dim=-1)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return x


class ContextAggregator(nn.Module):
    """
    Attention-based temporal context compression.
    """
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class FutureDecoder(nn.Module):
    """
    Projects global context onto learnable future query tokens.
    """
    def __init__(self, d_model: int = 64, forecast_horizon: int = FORECAST_HORIZON):
        super().__init__()
        self.future_queries = nn.Parameter(torch.randn(1, forecast_horizon, d_model))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        batch_size = context.size(0)
        future_queries = self.future_queries.repeat(batch_size, 1, 1)
        decoder_input = future_queries + context.unsqueeze(1)
        return decoder_input


class ProbabilisticHead(nn.Module):
    """
    Outputs predictive mean (mu) and variance/std (sigma) parameters.
    """
    def __init__(self, d_model: int = 64, output_dim: int = len(TARGET_COLS_4)):
        super().__init__()
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim * 2)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        out = self.network(x)
        mu = out[:, :, : self.output_dim]
        sigma = torch.clamp(F.softplus(out[:, :, self.output_dim :]), min=0.05, max=2.0)
        return mu, sigma


class SpikeHead(nn.Module):
    """
    Binary classification head predicting probability of anomalous perturbation/spike.
    """
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.network(x)
        return torch.sigmoid(logits).squeeze(-1)


class GNSSForecaster(nn.Module):
    """
    Complete Multi-Task Deep Transformer Forecaster for GNSS Ephemeris & Clock Errors.
    """
    def __init__(
        self,
        num_features: int,
        num_satellites: int,
        d_model: int = 64,
        forecast_horizon: int = FORECAST_HORIZON,
        output_dim: int = len(TARGET_COLS_4),
        embedding_dim: int = 8,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = GNSSTransformerBackbone(
            num_features=num_features,
            num_satellites=num_satellites,
            d_model=d_model,
            embedding_dim=embedding_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
        self.aggregator = ContextAggregator(d_model=d_model)
        self.future_decoder = FutureDecoder(d_model=d_model, forecast_horizon=forecast_horizon)
        self.prob_head = ProbabilisticHead(d_model=d_model, output_dim=output_dim)
        self.spike_head = SpikeHead(d_model=d_model)

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor) -> tuple:
        latent = self.backbone(x, sat_ids)
        context = self.aggregator(latent)
        decoder_input = self.future_decoder(context)
        spike_probs = self.spike_head(decoder_input)

        # Spike conditioning
        spike_context = spike_probs.unsqueeze(-1)
        conditioned_decoder_input = decoder_input * (1.0 + 0.25 * spike_context)

        mu, sigma = self.prob_head(conditioned_decoder_input)
        return mu, sigma, spike_probs, context

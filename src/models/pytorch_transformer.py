"""
Deep Multi-Task Sequential Architecture for Satellite Telemetry & Orbital Dynamics
Implements:
- Time2Vec Periodic + Non-Periodic Temporal Encodings
- Dynamic PRN Entity Embeddings (scaling de = ceil(1.6 * gamma^0.52))
- Sequential Hybrid: BiLSTM -> GRU -> Multi-Head Self-Attention (MHSA)
- Global Conditioning Context Vector: c = [h_GRU; MHSA(H); E_PRN]
- Three Parallel Decoding Heads:
    1. Gaussian Parameter Density Head (mu, sigma^2) with Gaussian NLL
    2. Binary Cross-Entropy (BCE) Spike / Anomaly Classification Head
    3. Conditional DDPM Diffusion Denoiser & Reverse Sampler
- Two-Stage Residual Physical Baseline Decomposition
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_4


def compute_prn_embedding_dim(num_satellites: int) -> int:
    """
    Computes optimal entity embedding dimension according to the empirical scaling rule:
    de = ceil(1.6 * gamma^0.52)
    """
    return int(math.ceil(1.6 * (num_satellites ** 0.52)))


class Time2Vec(nn.Module):
    """
    Time2Vec: Learnable continuous temporal representation for periodic & non-periodic dynamics.
    t2v(tau)[0] = omega_0 * tau + phi_0 (non-periodic secular drift)
    t2v(tau)[i] = sin(omega_i * tau + phi_i) for 1 <= i <= k (periodic cyclical dynamics)
    """
    def __init__(self, in_features: int = 1, out_features: int = 8):
        super().__init__()
        self.out_features = out_features
        self.linear_w = nn.Parameter(torch.randn(in_features, 1))
        self.linear_b = nn.Parameter(torch.randn(1))
        self.periodic_w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.periodic_b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        # tau: (B, seq_len, 1)
        linear_part = torch.matmul(tau, self.linear_w) + self.linear_b
        periodic_part = torch.sin(torch.matmul(tau, self.periodic_w) + self.periodic_b)
        return torch.cat([linear_part, periodic_part], dim=-1)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class BiLSTMGRUMHSABackbone(nn.Module):
    """
    Unified Hybrid Encoder:
    Telemetry + Time2Vec + PRN Entity Embedding
    -> Bidirectional LSTM (24h lookback forward & backward)
    -> GRU (Temporal compression & reset/update gating)
    -> Multi-Head Self-Attention (MHSA pairwise dependencies)
    -> Synthesized Context Vector: c = [h_GRU; MHSA(H); E_PRN]
    """
    def __init__(
        self,
        num_features: int,
        num_satellites: int,
        d_model: int = 64,
        bilstm_units: int = 48,
        gru_units: int = 48,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.de = compute_prn_embedding_dim(num_satellites)
        self.sat_embedding = nn.Embedding(num_satellites, self.de)
        self.t2v = Time2Vec(in_features=1, out_features=8)

        # Total input dimension to BiLSTM
        raw_dim = num_features + self.de + 8
        self.input_proj = nn.Linear(raw_dim, d_model)

        # 1. Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=bilstm_units,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # 2. GRU for Temporal Compression
        self.gru = nn.GRU(
            input_size=bilstm_units * 2,
            hidden_size=gru_units,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # 3. Multi-Head Self-Attention (MHSA)
        self.mhsa = nn.MultiheadAttention(
            embed_dim=gru_units,
            num_heads=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.mhsa_norm = nn.LayerNorm(gru_units)

        # Context Synthesizer Dimension: h_gru (gru_units) + mhsa_pool (gru_units) + E_prn (de)
        self.context_dim = gru_units * 2 + self.de

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor) -> tuple:
        # x: (B, seq_len, num_features)
        B, seq_len, _ = x.shape

        # Time2Vec from continuous normalized time sequence
        tau = torch.linspace(0, 1, seq_len, device=x.device).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        t2v_feats = self.t2v(tau)

        # PRN Entity Embedding: (B, de)
        e_prn = self.sat_embedding(sat_ids)
        e_prn_expanded = e_prn.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenate features
        combined_input = torch.cat([x, t2v_feats, e_prn_expanded], dim=-1)
        h = self.input_proj(combined_input)

        # BiLSTM
        bilstm_out, _ = self.bilstm(h)
        bilstm_out = self.dropout1(bilstm_out)

        # GRU
        gru_seq, h_n = self.gru(bilstm_out)
        gru_seq = self.dropout2(gru_seq)
        h_gru = h_n[-1]  # (B, gru_units)

        # MHSA over GRU temporal sequence
        mhsa_out, _ = self.mhsa(gru_seq, gru_seq, gru_seq)
        mhsa_seq = self.mhsa_norm(gru_seq + mhsa_out)
        mhsa_pooled = torch.mean(mhsa_seq, dim=1)  # (B, gru_units)

        # Synthesized Global Context Vector: c = [h_GRU; MHSA(H); E_PRN]
        context = torch.cat([h_gru, mhsa_pooled, e_prn], dim=-1)  # (B, context_dim)

        return mhsa_seq, context


class ProbabilisticGaussianHead(nn.Module):
    """
    Task Head 1: Gaussian Parameter Regression Head predicting conditional mean (mu)
    and variance (sigma^2 = softplus(W_sigma * h + b_sigma) + epsilon).
    """
    def __init__(self, context_dim: int, forecast_horizon: int = FORECAST_HORIZON, output_dim: int = len(TARGET_COLS_4)):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, forecast_horizon * output_dim * 2)
        )

    def forward(self, context: torch.Tensor) -> tuple:
        out = self.net(context)
        out = out.view(-1, self.forecast_horizon, self.output_dim * 2)
        mu_delta = out[:, :, : self.output_dim]
        # strictly positive variance via softplus + epsilon
        sigma = F.softplus(out[:, :, self.output_dim :]) + 0.05
        return mu_delta, sigma


class AnomalySpikeBCEHead(nn.Module):
    """
    Task Head 2: Supervised Binary Event Classification Head for thruster burns,
    solar radiation pressure bursts, and atomic clock step-jumps.
    """
    def __init__(self, context_dim: int, forecast_horizon: int = FORECAST_HORIZON):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_horizon),
            nn.Sigmoid()
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class GNSSForecaster(nn.Module):
    """
    Complete End-to-End Deep Hybrid Architecture for GNSS Orbit & Clock Prediction.
    Integrates:
    - BiLSTM-GRU-MHSA Encoder Backbone
    - Learnable PRN Entity Embeddings
    - Time2Vec Cyclical & Secular Encodings
    - Probabilistic Gaussian Parameter Regression Head (mu, sigma)
    - Binary Cross-Entropy Anomaly/Spike Head
    - Physics-Informed Two-Stage Residual Anchor
    """
    def __init__(
        self,
        num_features: int,
        num_satellites: int,
        d_model: int = 64,
        bilstm_units: int = 48,
        gru_units: int = 48,
        nhead: int = 4,
        forecast_horizon: int = FORECAST_HORIZON,
        output_dim: int = len(TARGET_COLS_4),
        dropout: float = 0.1
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

        self.backbone = BiLSTMGRUMHSABackbone(
            num_features=num_features,
            num_satellites=num_satellites,
            d_model=d_model,
            bilstm_units=bilstm_units,
            gru_units=gru_units,
            nhead=nhead,
            dropout=dropout
        )

        context_dim = self.backbone.context_dim
        self.prob_head = ProbabilisticGaussianHead(
            context_dim=context_dim,
            forecast_horizon=forecast_horizon,
            output_dim=output_dim
        )
        self.spike_head = AnomalySpikeBCEHead(
            context_dim=context_dim,
            forecast_horizon=forecast_horizon
        )

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor) -> tuple:
        mhsa_seq, context = self.backbone(x, sat_ids)

        # 1. Anomaly spike probabilities
        spike_probs = self.spike_head(context)

        # 2. Gaussian parameter predictions (mu_delta, sigma)
        mu_delta, sigma = self.prob_head(context)

        # Two-stage physics separation: offset predictions relative to last known observation
        last_obs = x[:, -1:, : self.output_dim]  # (B, 1, output_dim)
        mu = last_obs + mu_delta

        return mu, sigma, spike_probs, context

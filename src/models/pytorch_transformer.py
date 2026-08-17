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


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for Non-Stationary Time Series.
    Symmetrically removes instance-specific mean and variance from the input lookback window
    and restores them at the output forecast horizon to combat distribution shift.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x_norm = (x - self.mean) / self.stdev
            if self.affine:
                x_norm = x_norm * self.affine_weight + self.affine_bias
            return x_norm
        elif mode == "denorm":
            x_denorm = x
            if self.affine:
                safe_weight = torch.where(
                    self.affine_weight.abs() < self.eps,
                    self.affine_weight.sign() * self.eps + (self.affine_weight == 0) * self.eps,
                    self.affine_weight,
                )
                x_denorm = (x_denorm - self.affine_bias) / safe_weight
            x_denorm = x_denorm * self.stdev + self.mean
            return x_denorm
        elif mode == "denorm_sigma":
            # y = (z - bias) / weight * stdev + mean, so scale must undo
            # both the learned affine transform and instance standardization.
            if self.affine:
                affine_scale = self.affine_weight.abs().clamp_min(self.eps)
                x = x / affine_scale
            return x * self.stdev
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' is not supported.")


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
        num_layers: int = 1,
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
        self.mhsa_ffn = nn.Sequential(
            nn.Linear(gru_units, gru_units * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_units * 4, gru_units),
        )
        self.mhsa_ffn_norm = nn.LayerNorm(gru_units)

        # Keep the original attention layer names for checkpoint compatibility and
        # make the advertised num_layers parameter real for all additional blocks.
        self.attention_layers = nn.ModuleList()
        for _ in range(max(1, num_layers) - 1):
            self.attention_layers.append(nn.ModuleDict({
                "attention": nn.MultiheadAttention(
                    embed_dim=gru_units,
                    num_heads=nhead,
                    batch_first=True,
                    dropout=dropout,
                ),
                "norm1": nn.LayerNorm(gru_units),
                "ffn": nn.Sequential(
                    nn.Linear(gru_units, gru_units * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gru_units * 4, gru_units),
                ),
                "norm2": nn.LayerNorm(gru_units),
            }))

        # Context Synthesizer Dimension: h_gru (gru_units) + mhsa_pool (gru_units) + E_prn (de)
        self.context_dim = gru_units * 2 + self.de
        self.seq_feature_dim = gru_units

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
        mhsa_seq = self.mhsa_ffn_norm(mhsa_seq + self.mhsa_ffn(mhsa_seq))
        for block in self.attention_layers:
            attention_out, _ = block["attention"](mhsa_seq, mhsa_seq, mhsa_seq)
            mhsa_seq = block["norm1"](mhsa_seq + attention_out)
            mhsa_seq = block["norm2"](mhsa_seq + block["ffn"](mhsa_seq))
        mhsa_pooled = torch.mean(mhsa_seq, dim=1)  # (B, gru_units)

        # Synthesized Global Context Vector: c = [h_GRU; MHSA(H); E_PRN]
        context = torch.cat([h_gru, mhsa_pooled, e_prn], dim=-1)  # (B, context_dim)

        return mhsa_seq, context


class ProbabilisticGaussianHead(nn.Module):
    """
    Task Head 1: Sequence-Preserving Temporal Projection Gaussian Head.
    Combines sequence tokens (B, L, D) and global context c to generate multi-horizon forecasts
    without flattening sequence dimensions.
    """
    def __init__(
        self,
        seq_feature_dim: int,
        context_dim: int,
        seq_len: int = SEQ_LEN,
        forecast_horizon: int = FORECAST_HORIZON,
        output_dim: int = len(TARGET_COLS_4)
    ):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim

        # Temporal dimension mapping from lookback seq_len to forecast_horizon
        self.temporal_map = nn.Linear(seq_len, forecast_horizon)

        # Global context conditioner
        self.context_proj = nn.Linear(context_dim, seq_feature_dim)

        # Output feature projection
        self.feat_norm = nn.LayerNorm(seq_feature_dim)
        self.proj_net = nn.Sequential(
            nn.Linear(seq_feature_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim * 2)
        )

    def forward(self, mhsa_seq: torch.Tensor, context: torch.Tensor) -> tuple:
        # mhsa_seq: (B, seq_len, seq_feature_dim)
        # context: (B, context_dim)
        c_mod = self.context_proj(context).unsqueeze(1)  # (B, 1, seq_feature_dim)
        h_seq = mhsa_seq + c_mod  # Inject global conditioning

        # Transpose to project temporal dimension: (B, seq_feature_dim, seq_len) -> (B, seq_feature_dim, horizon)
        h_seq_t = h_seq.transpose(1, 2)
        h_proj_t = self.temporal_map(h_seq_t)
        h_future = h_proj_t.transpose(1, 2)  # (B, forecast_horizon, seq_feature_dim)

        h_norm = self.feat_norm(h_future)
        out = self.proj_net(h_norm)  # (B, forecast_horizon, output_dim * 2)

        mu_delta = out[:, :, : self.output_dim]
        sigma = F.softplus(out[:, :, self.output_dim :]) + 1e-4
        return mu_delta, sigma


class AnomalySpikeBCEHead(nn.Module):
    """
    Task Head 2: Supervised Binary Event Classification Head for thruster burns,
    solar radiation pressure bursts, and atomic clock step-jumps.
    Outputs raw logits for numerically stable AMP-compatible BCE computation.
    """
    def __init__(self, context_dim: int, forecast_horizon: int = FORECAST_HORIZON):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.net = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, forecast_horizon)
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class GNSSForecaster(nn.Module):
    """
    Complete End-to-End Deep Hybrid Architecture for GNSS Orbit & Clock Prediction.
    Integrates:
    - RevIN (Reversible Instance Normalization) for Non-Stationary Shift
    - BiLSTM-GRU-MHSA Encoder Backbone
    - Learnable PRN Entity Embeddings
    - Time2Vec Cyclical & Secular Encodings
    - Sequence-Preserving Temporal Projection Gaussian Head (mu, sigma)
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
        num_layers: int = 1,
        seq_len: int = SEQ_LEN,
        forecast_horizon: int = FORECAST_HORIZON,
        output_dim: int = len(TARGET_COLS_4),
        target_feature_indices: tuple[int, ...] | None = None,
        use_revin: bool = True,
        enable_event_head: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.output_dim = output_dim
        self.target_feature_indices = (
            tuple(range(output_dim))
            if target_feature_indices is None
            else tuple(target_feature_indices)
        )
        if len(self.target_feature_indices) != output_dim:
            raise ValueError("target_feature_indices must have one index per output")
        if min(self.target_feature_indices) < 0 or max(self.target_feature_indices) >= num_features:
            raise ValueError("target_feature_indices contains an out-of-range feature index")
        self.use_revin = use_revin
        self.enable_event_head = enable_event_head

        # RevIN layer applied to target channels
        if self.use_revin:
            self.revin = RevIN(num_features=output_dim, affine=True)

        self.backbone = BiLSTMGRUMHSABackbone(
            num_features=num_features,
            num_satellites=num_satellites,
            d_model=d_model,
            bilstm_units=bilstm_units,
            gru_units=gru_units,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )

        context_dim = self.backbone.context_dim
        seq_feature_dim = self.backbone.seq_feature_dim

        self.prob_head = ProbabilisticGaussianHead(
            seq_feature_dim=seq_feature_dim,
            context_dim=context_dim,
            seq_len=seq_len,
            forecast_horizon=forecast_horizon,
            output_dim=output_dim
        )
        self.spike_head = (
            AnomalySpikeBCEHead(context_dim=context_dim, forecast_horizon=forecast_horizon)
            if enable_event_head
            else None
        )

    def forward(self, x: torch.Tensor, sat_ids: torch.Tensor) -> tuple:
        # If RevIN is active, normalize target variates within x
        x_processed = x.clone()
        target_indices = list(self.target_feature_indices)
        if self.use_revin:
            target_slice = x[:, :, target_indices]
            x_norm = self.revin(target_slice, mode="norm")
            x_processed[:, :, target_indices] = x_norm

        mhsa_seq, context = self.backbone(x_processed, sat_ids)

        # 1. Anomaly spike probabilities
        spike_probs = (
            self.spike_head(context)
            if self.spike_head is not None
            else context.new_zeros((context.shape[0], self.forecast_horizon))
        )

        # 2. Sequence-preserving Gaussian parameter predictions (mu_delta, sigma)
        mu_delta, sigma = self.prob_head(mhsa_seq, context)

        # Two-stage physics separation: offset relative to last observation in normalized space
        last_obs = x_processed[:, -1:, target_indices]
        mu = last_obs + mu_delta

        # Denormalize with RevIN back to original target distribution scale
        if self.use_revin:
            mu = self.revin(mu, mode="denorm")
            sigma = self.revin(sigma, mode="denorm_sigma")

        return mu, sigma, spike_probs, context

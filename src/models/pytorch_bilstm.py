"""
Enhanced PyTorch Implementation of the Multi-Horizon BiLSTM + GRU Architecture
Incorporates Residual Skip Anchor, Attention Context Pooling, and LayerNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_5


class BiLSTMGRUPyTorchModel(nn.Module):
    """
    Enhanced Direct Multi-Step BiLSTM + GRU Forecaster with Residual Anchor Connection.

    Architecture:
        Input (B, seq_len, n_features)
        -> Bidirectional LSTM (bilstm_units, batch_first=True)
        -> Dropout(dropout_1)
        -> GRU (gru_units, batch_first=True)
        -> Attention Pooling + Last Hidden State Concat
        -> LayerNorm + Dense Projection Head
        -> Residual Anchor Connection (+ x[:, -1:, :])
        -> Output (B, forecast_horizon, n_features)
    """
    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        n_features: int = len(TARGET_COLS_5),
        forecast_horizon: int = FORECAST_HORIZON,
        bilstm_units: int = 64,
        gru_units: int = 64,
        dropout_1: float = 0.2,
        dropout_2: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.forecast_horizon = forecast_horizon

        # Bidirectional LSTM: outputs (B, seq_len, 2 * bilstm_units)
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=bilstm_units,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_1)

        # GRU: input is (B, seq_len, 2 * bilstm_units), outputs (B, seq_len, gru_units)
        self.gru = nn.GRU(
            input_size=bilstm_units * 2,
            hidden_size=gru_units,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_2)

        # Attention Pooling over all recurrent steps
        self.attn_pool = nn.Sequential(
            nn.Linear(gru_units, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.layer_norm = nn.LayerNorm(gru_units * 2)

        # Non-linear Dense Projection Head
        self.dense_proj = nn.Sequential(
            nn.Linear(gru_units * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, forecast_horizon * n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout1(lstm_out)

        gru_out, _ = self.gru(lstm_out)
        gru_out = self.dropout2(gru_out)

        # Last timestep hidden state
        last_state = gru_out[:, -1, :]  # (B, gru_units)

        # Attention weighted context pooling across all timesteps
        attn_weights = torch.softmax(self.attn_pool(gru_out), dim=1)  # (B, seq_len, 1)
        attn_context = torch.sum(attn_weights * gru_out, dim=1)        # (B, gru_units)

        # Combined representation
        combined = torch.cat([last_state, attn_context], dim=-1)      # (B, gru_units * 2)
        combined = self.layer_norm(combined)

        # Direct multi-step delta/residual prediction
        delta = self.dense_proj(combined)
        delta = delta.view(-1, self.forecast_horizon, self.n_features)

        # Residual Skip Anchor: prediction is offset relative to last observed timestep
        last_obs = x[:, -1:, :]  # (B, 1, n_features)
        out = last_obs + delta

        return out

"""
PyTorch Implementation of the Multi-Horizon BiLSTM + GRU Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_5


class BiLSTMGRUPyTorchModel(nn.Module):
    """
    Direct Multi-Step BiLSTM + GRU Forecaster in PyTorch.

    Architecture:
        Input (B, seq_len, n_features)
        -> Bidirectional LSTM (bilstm_units, batch_first=True)
        -> Dropout(dropout_1)
        -> GRU (gru_units, batch_first=True) [takes last hidden state]
        -> Dropout(dropout_2)
        -> LayerNorm(gru_units)
        -> Linear(gru_units, 64) -> ReLU
        -> Linear(64, forecast_horizon * n_features)
        -> Reshape to (B, forecast_horizon, n_features)
    """
    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        n_features: int = len(TARGET_COLS_5),
        forecast_horizon: int = FORECAST_HORIZON,
        bilstm_units: int = 32,
        gru_units: int = 64,
        dropout_1: float = 0.3,
        dropout_2: float = 0.11
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
        self.layer_norm = nn.LayerNorm(gru_units)

        # Dense Projection Head
        self.dense_proj = nn.Sequential(
            nn.Linear(gru_units, 64),
            nn.ReLU(),
            nn.Linear(64, forecast_horizon * n_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_features)
        lstm_out, _ = self.bilstm(x)
        lstm_out = self.dropout1(lstm_out)

        gru_out, h_n = self.gru(lstm_out)
        # Take the last timestep's output: (B, gru_units)
        context = gru_out[:, -1, :]
        context = self.dropout2(context)
        context = self.layer_norm(context)

        out = self.dense_proj(context)
        out = out.view(-1, self.forecast_horizon, self.n_features)
        return out

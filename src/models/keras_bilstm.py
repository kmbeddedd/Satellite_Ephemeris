"""
TensorFlow / Keras Multi-Horizon BiLSTM + GRU Architecture
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout,
    Bidirectional, LayerNormalization, Reshape
)
from tensorflow.keras.optimizers import Adam
from src.config import SEQ_LEN, FORECAST_HORIZON, TARGET_COLS_5


def build_bilstm_gru_model(
    seq_len: int = SEQ_LEN,
    n_features: int = len(TARGET_COLS_5),
    forecast_horizon: int = FORECAST_HORIZON,
    bilstm_units: int = 32,
    gru_units: int = 64,
    dropout_1: float = 0.3,
    dropout_2: float = 0.11,
    learning_rate: float = 1.56e-3,
    loss: str = "huber"
) -> Model:
    """
    Constructs and compiles the BiLSTM + GRU Multi-Horizon Direct Forecaster.

    Architecture:
        Input (seq_len, n_features)
        -> Bidirectional LSTM (bilstm_units, return_sequences=True)
        -> Dropout(dropout_1)
        -> GRU (gru_units, return_sequences=False)
        -> Dropout(dropout_2)
        -> LayerNormalization
        -> Dense(64, activation='relu')
        -> Dense(forecast_horizon * n_features)
        -> Reshape((forecast_horizon, n_features))
    """
    inp = Input(shape=(seq_len, n_features), name="input")

    # Bidirectional LSTM Encoder
    x = Bidirectional(LSTM(bilstm_units, return_sequences=True), name="bilstm")(inp)
    x = Dropout(dropout_1, name="dropout_1")(x)

    # GRU Bottleneck
    x = GRU(gru_units, return_sequences=False, name="gru")(x)
    x = Dropout(dropout_2, name="dropout_2")(x)
    x = LayerNormalization(name="layer_norm")(x)

    # Projection Head
    x = Dense(64, activation="relu", name="dense_proj")(x)
    out = Dense(forecast_horizon * n_features, name="dense_out")(x)
    out = Reshape((forecast_horizon, n_features), name="output")(out)

    model = Model(inputs=inp, outputs=out, name="GNSS_BiLSTM_GRU_Forecaster")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["mae"]
    )
    return model

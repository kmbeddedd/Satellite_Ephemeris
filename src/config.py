"""
Global Configuration and Constants for GNSS Orbit & Clock Forecasting
"""

import os

# Default Paths
DEFAULT_DATA_PATH = "data_acquisition/CLEAN_GNSS_BENCHMARK.csv"
DEFAULT_OUTPUT_DIR = "./gnss_results"
ORBITIQ_DATA_PATH = "data/orbitiq/ORBITIQ_ISRO_BENCHMARK.csv"
ORBITIQ_OUTPUT_DIR = "./orbitiq_pipeline_results"

# Temporal Specifications
TRAIN_END_DATE = "2026-01-08 00:00:00"  # Day 1-7 Train, Day 8 Test
TOTAL_TIMESTEPS_PER_SAT = 768            # 8 days * 96 steps/day (15-min interval)
SEQ_LEN = 96                            # Lookback window = 24 hours (96 steps)
FORECAST_HORIZON = 96                   # Direct multi-step prediction = 24 hours (96 steps)

# Data diagnostics. Large targets are reported but never filtered using future truth.
OUTLIER_THRESHOLD_3D = 50_000.0
# Retained only for the backward-compatible SPIKE arrays. Pseudo-events derived
# from target magnitude are not used by either corrected trainer.
SPIKE_THRESHOLD = 1.5
SP3_CLOCK_SENTINEL_SECONDS = 0.999999999999
EXPECTED_CADENCE_MINUTES = 15

# Target Variables
# Deprecated compatibility schema. 3D_Orbit_Error is deterministic and must not
# be learned independently; corrected trainers use TARGET_COLS_4 and derive 3D.
TARGET_COLS_5 = [
    "Error_X",
    "Error_Y",
    "Error_Z",
    "3D_Orbit_Error",
    "Error_Clock"
]

# 4-target coordinate set (used in PyTorch Transformer/Diffusion)
TARGET_COLS_4 = [
    "Error_X",
    "Error_Y",
    "Error_Z",
    "Error_Clock"
]

# Engineered Feature Columns
FEATURE_COLS_PYTORCH = [
    "Error_X",
    "Error_Y",
    "Error_Z",
    "Error_Clock",
    "time_sin",
    "time_cos",
    "Error_X_roll_mean",
    "Error_Y_roll_mean",
    "Error_Z_roll_mean",
    "Error_Clock_roll_mean"
]

# Multi-Horizon Evaluation Intervals (Timestep mapping at 15-minute intervals)
HORIZON_MAP = {
    "15 min": 1,
    "30 min": 2,
    "1 hour": 4,
    "2 hours": 8,
    "6 hours": 24,
    "12 hours": 48,
    "24 hours": 96
}

# Model Hyperparameter Defaults
DEFAULT_SEED = 42

# Transformer Defaults (PyTorch)
TRANSFORMER_DEFAULTS = {
    "epochs": 30,
    "batch_size": 32,
    "d_model": 64,
    "embedding_dim": 8,
    "nhead": 4,
    "num_layers": 3,
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "lr_patience": 3
}

# Diffusion Defaults (PyTorch)
DIFFUSION_DEFAULTS = {
    "epochs": 80,
    "steps": 100,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5
}

"""
Data Processing, Feature Engineering, and Dataset Builders for GNSS Forecasting
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional

from src.config import (
    TRAIN_END_DATE,
    TOTAL_TIMESTEPS_PER_SAT,
    SEQ_LEN,
    FORECAST_HORIZON,
    OUTLIER_THRESHOLD_3D,
    SPIKE_THRESHOLD,
    TARGET_COLS_5,
    TARGET_COLS_4,
    FEATURE_COLS_PYTORCH,
    DEFAULT_SEED
)


def load_and_clean_data(
    data_path: str,
    outlier_threshold: float = OUTLIER_THRESHOLD_3D,
    train_end_date: str = TRAIN_END_DATE
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads GNSS dataset, sorts by satellite and timestamp,
    filters incomplete satellite series, removes gross physical outliers,
    and partitions into train (Day 1-7) and test (Day 8) splits.
    """
    print("=" * 65)
    print("Loading & Preprocessing GNSS Telemetry Data")
    print("=" * 65)

    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)

    # Filter complete satellites with 768 timesteps (8 days * 96 steps)
    sat_counts = df.groupby("Satellite_ID").size()
    complete_sats = sorted(sat_counts[sat_counts == TOTAL_TIMESTEPS_PER_SAT].index.tolist())
    df = df[df["Satellite_ID"].isin(complete_sats)].copy()

    # Filter physical outliers (e.g. initialisation spikes > 50 km)
    if "3D_Orbit_Error" in df.columns:
        df = df[df["3D_Orbit_Error"] < outlier_threshold].copy()

    # Train (Day 1-7) and Test (Day 8) Split
    train_df = df[df["Timestamp"] < pd.Timestamp(train_end_date)].copy()
    test_df = df[df["Timestamp"] >= pd.Timestamp(train_end_date)].copy()

    gps_sats = [s for s in complete_sats if s.startswith("G")]
    glo_sats = [s for s in complete_sats if s.startswith("R")]

    print(f"  Complete Satellites : {len(complete_sats)} (GPS={len(gps_sats)}, GLONASS={len(glo_sats)})")
    print(f"  Training Records    : {len(train_df):,} rows (Day 1-7)")
    print(f"  Testing Records     : {len(test_df):,} rows (Day 8)")

    return train_df, test_df, complete_sats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes diurnal cyclical time features (sin/cos of daily minute)
    and per-satellite rolling statistics (rolling mean window=8).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)

    # Cyclical daily time representations (1440 minutes in a day)
    time_minutes = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    df["time_sin"] = np.sin(2 * np.pi * time_minutes / 1440.0)
    df["time_cos"] = np.cos(2 * np.pi * time_minutes / 1440.0)

    # Rolling window mean features per satellite
    roll_cols = ["Error_X", "Error_Y", "Error_Z", "Error_Clock"]
    for col in roll_cols:
        if col in df.columns:
            df[f"{col}_roll_mean"] = (
                df.groupby("Satellite_ID")[col]
                .transform(lambda x: x.rolling(8, min_periods=1).mean())
            )

    return df


def scale_datasets_keras(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: List[str] = TARGET_COLS_5
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scales targets using StandardScaler fitted strictly on training data.
    """
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[target_cols] = scaler.fit_transform(train_df[target_cols])
    test_df[target_cols] = scaler.transform(test_df[target_cols])

    return train_df, test_df, scaler


def make_sliding_sequences(
    series: np.ndarray,
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_HORIZON
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs (N, seq_len, F) lookback and (N, horizon, F) target sequences.
    """
    X, y = [], []
    num_samples = len(series) - seq_len - horizon + 1
    for i in range(num_samples):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len : i + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_keras_sequences(
    train_df: pd.DataFrame,
    complete_sats: List[str],
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_HORIZON,
    target_cols: List[str] = TARGET_COLS_5,
    val_split: float = 0.1,
    seed: int = DEFAULT_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates training and validation sequence tensors across all complete satellites.
    """
    np.random.seed(seed)
    X_all, y_all = [], []

    for sat_id in complete_sats:
        sat_data = train_df[train_df["Satellite_ID"] == sat_id][target_cols].values
        X_s, y_s = make_sliding_sequences(sat_data, seq_len, horizon)
        if len(X_s) > 0:
            X_all.append(X_s)
            y_all.append(y_s)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Shuffle sequences so validation covers all satellites evenly
    indices = np.random.permutation(len(X_all))
    X_all, y_all = X_all[indices], y_all[indices]

    split_idx = int((1.0 - val_split) * len(X_all))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]

    print(f"  Training Sequences   : {X_train.shape}")
    print(f"  Validation Sequences : {X_val.shape}")

    return X_train, y_train, X_val, y_val


def prepare_pytorch_datasets(
    data_path: str,
    input_window: int = SEQ_LEN,
    forecast_horizon: int = FORECAST_HORIZON,
    spike_threshold: float = SPIKE_THRESHOLD,
    batch_size: int = 32,
    test_size: float = 0.3,
    val_ratio: float = 0.5
) -> Dict:
    """
    Builds PyTorch Datasets & DataLoaders with engineered features,
    satellite entity encoding, target scaling, and anomaly spike labels.
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split

    class GNSSPyTorchDataset(Dataset):
        def __init__(self, X, Y, SAT, SPIKE):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.Y = torch.tensor(Y, dtype=torch.float32)
            self.SAT = torch.tensor(SAT, dtype=torch.long)
            self.SPIKE = torch.tensor(SPIKE, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx], self.SAT[idx], self.SPIKE[idx]

    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)

    sat_encoder = LabelEncoder()
    df["sat_idx"] = sat_encoder.fit_transform(df["Satellite_ID"])

    # Feature engineering
    df = engineer_features(df)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    feature_cols = FEATURE_COLS_PYTORCH
    target_cols = TARGET_COLS_4

    df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
    df[target_cols] = target_scaler.fit_transform(df[target_cols])

    X, Y, SAT, SPIKE = [], [], [], []

    for sat_id in df["sat_idx"].unique():
        sat_df = df[df["sat_idx"] == sat_id]
        feat_vals = sat_df[feature_cols].values
        targ_vals = sat_df[target_cols].values
        num_windows = len(sat_df) - input_window - forecast_horizon

        for i in range(num_windows):
            x_seq = feat_vals[i : i + input_window]
            y_seq = targ_vals[i + input_window : i + input_window + forecast_horizon]
            spike_labels = (np.abs(y_seq) > spike_threshold).any(axis=1).astype(np.float32)

            X.append(x_seq)
            Y.append(y_seq)
            SAT.append(sat_id)
            SPIKE.append(spike_labels)

    X = np.array(X)
    Y = np.array(Y)
    SAT = np.array(SAT)
    SPIKE = np.array(SPIKE)

    # Train / Temp split
    X_train, X_temp, Y_train, Y_temp, SAT_train, SAT_temp, SPIKE_train, SPIKE_temp = train_test_split(
        X, Y, SAT, SPIKE, test_size=test_size, shuffle=False
    )

    # Val / Test split
    X_val, X_test, Y_val, Y_test, SAT_val, SAT_test, SPIKE_val, SPIKE_test = train_test_split(
        X_temp, Y_temp, SAT_temp, SPIKE_temp, test_size=val_ratio, shuffle=False
    )

    train_ds = GNSSPyTorchDataset(X_train, Y_train, SAT_train, SPIKE_train)
    val_ds = GNSSPyTorchDataset(X_val, Y_val, SAT_val, SPIKE_val)
    test_ds = GNSSPyTorchDataset(X_test, Y_test, SAT_test, SPIKE_test)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sat_encoder": sat_encoder,
        "num_features": len(feature_cols),
        "num_satellites": len(sat_encoder.classes_),
        "output_dim": len(target_cols),
        "X_train": X_train,
        "Y_train": Y_train,
        "SAT_train": SAT_train,
        "SPIKE_train": SPIKE_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "SAT_val": SAT_val,
        "SPIKE_val": SPIKE_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "SAT_test": SAT_test,
        "SPIKE_test": SPIKE_test
    }


class FastGPUTensorLoader:
    """
    Zero-overhead direct GPU VRAM Tensor Loader.
    Keeps all data resident in GPU VRAM memory and slices mini-batches purely on CUDA.
    Completely eliminates CPU-to-GPU transfer bottleneck and maximizes GPU SM utilization.
    """
    def __init__(self, tensors: tuple, batch_size: int = 128, shuffle: bool = True, device: torch.device = None):
        self.tensors = [t.to(device) if device is not None else t for t in tensors]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.tensors[0])
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.device = device or self.tensors[0].device

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples, device=self.device)
        else:
            indices = torch.arange(self.num_samples, device=self.device)

        for i in range(self.num_batches):
            batch_idx = indices[i * self.batch_size : (i + 1) * self.batch_size]
            yield tuple(t[batch_idx] for t in self.tensors)

    def __len__(self):
        return self.num_batches


"""Leakage-safe data processing and dataset builders for GNSS forecasting.

Rows are never removed because of a future target value. Missing SP3 clocks are
represented by masks, accepted windows have exact 15-minute cadence, temporal
split labels are disjoint, and scalers are fitted on the training block only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    DEFAULT_SEED,
    FEATURE_COLS_PYTORCH,
    FORECAST_HORIZON,
    OUTLIER_THRESHOLD_3D,
    SEQ_LEN,
    SPIKE_THRESHOLD,
    TARGET_COLS_4,
    TRAIN_END_DATE,
)


EXPECTED_INTERVAL = pd.Timedelta(minutes=15)
SP3_CLOCK_SENTINEL_SECONDS = 0.999999999999
SP3_CLOCK_SENTINEL_ATOL = 1e-9
VALID_SUFFIX = "_valid"
PHYSICAL_FEATURE_COLS = [
    "Broadcast_X",
    "Broadcast_Y",
    "Broadcast_Z",
    "Broadcast_Clock",
    "Broadcast_VX",
    "Broadcast_VY",
    "Broadcast_VZ",
    "Broadcast_Clock_Drift",
    "Broadcast_Radius",
    "Broadcast_Phase_Sin",
    "Broadcast_Phase_Cos",
]


def _as_interval(interval: str | pd.Timedelta) -> pd.Timedelta:
    value = pd.Timedelta(interval)
    if value <= pd.Timedelta(0):
        raise ValueError(f"interval must be positive, got {interval!r}")
    return value


def _validate_target_columns(target_cols: Sequence[str]) -> List[str]:
    columns = list(target_cols)
    if not columns:
        raise ValueError("At least one target column is required.")
    if len(columns) != len(set(columns)):
        raise ValueError(f"Target columns contain duplicates: {columns}")
    unsupported = sorted(set(columns) - set(TARGET_COLS_4))
    if unsupported:
        raise ValueError(
            "Only Error_X, Error_Y, Error_Z, and Error_Clock may be learned "
            f"targets; unsupported targets: {unsupported}. Derive 3D orbit error "
            "from predicted coordinates during evaluation."
        )
    return columns


def _clock_sentinel_mask(df: pd.DataFrame) -> pd.Series:
    """Return rows whose precise/SP3 clock field contains its missing sentinel."""
    sentinel = pd.Series(False, index=df.index, dtype=bool)
    source_found = False
    for column in ("Modelled_Clock", "Precise_Clock", "SP3_Clock"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            sentinel |= np.isclose(
                values,
                SP3_CLOCK_SENTINEL_SECONDS,
                rtol=0.0,
                atol=SP3_CLOCK_SENTINEL_ATOL,
            )
            source_found = True

    # Error_Clock = Broadcast_Clock - Precise_Clock in this project.
    if not source_found and {"Broadcast_Clock", "Error_Clock"}.issubset(df.columns):
        broadcast = pd.to_numeric(df["Broadcast_Clock"], errors="coerce")
        error = pd.to_numeric(df["Error_Clock"], errors="coerce")
        sentinel |= np.isclose(
            broadcast - error,
            SP3_CLOCK_SENTINEL_SECONDS,
            rtol=0.0,
            atol=SP3_CLOCK_SENTINEL_ATOL,
        )
        source_found = True

    # Support a raw SP3 clock series supplied directly under the target name.
    if not source_found and "Error_Clock" in df.columns:
        values = pd.to_numeric(df["Error_Clock"], errors="coerce")
        sentinel |= np.isclose(
            np.abs(values),
            SP3_CLOCK_SENTINEL_SECONDS,
            rtol=0.0,
            atol=SP3_CLOCK_SENTINEL_ATOL,
        )
    return sentinel


def apply_target_validity_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-target validity masks and replace invalid targets with NaN."""
    result = df.copy()
    missing = [column for column in TARGET_COLS_4 if column not in result.columns]
    if missing:
        raise ValueError(f"Dataset is missing required target columns: {missing}")

    sentinel = _clock_sentinel_mask(result)
    result["SP3_Clock_Sentinel"] = sentinel.to_numpy(dtype=bool)
    for column in TARGET_COLS_4:
        numeric = pd.to_numeric(result[column], errors="coerce")
        array = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        valid = np.isfinite(array)
        if column == "Error_Clock":
            valid &= ~sentinel.to_numpy(dtype=bool)
        result[f"{column}{VALID_SUFFIX}"] = valid
        result[column] = numeric.astype(float)
        result.loc[~valid, column] = np.nan
    return result


def cadence_report(
    df: pd.DataFrame,
    interval: str | pd.Timedelta = EXPECTED_INTERVAL,
) -> Dict[str, object]:
    """Summarize duplicate epochs and non-contiguous steps per satellite."""
    expected = _as_interval(interval)
    missing = sorted({"Timestamp", "Satellite_ID"} - set(df.columns))
    if missing:
        raise ValueError(f"Cannot validate cadence; missing columns: {missing}")
    timestamps = pd.to_datetime(df["Timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(
            f"Timestamp contains {int(timestamps.isna().sum())} unparsable/null values."
        )

    working = df.loc[:, ["Satellite_ID"]].copy()
    working["Timestamp"] = timestamps
    duplicates = int(working.duplicated(["Satellite_ID", "Timestamp"]).sum())
    irregular_by_satellite: Dict[str, int] = {}
    for satellite_id, sat_df in working.groupby("Satellite_ID", sort=True):
        diffs = sat_df.sort_values("Timestamp")["Timestamp"].diff().dropna()
        count = int((diffs != expected).sum())
        if count:
            irregular_by_satellite[str(satellite_id)] = count
    return {
        "interval_minutes": float(expected / pd.Timedelta(minutes=1)),
        "duplicate_epochs": duplicates,
        "irregular_steps": int(sum(irregular_by_satellite.values())),
        "irregular_by_satellite": irregular_by_satellite,
    }


def _read_data_frame(data_path: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = pd.read_csv(data_path)
    required = {"Timestamp", "Satellite_ID", *TARGET_COLS_4}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].isna().any():
        raise ValueError(
            f"Timestamp contains {int(df['Timestamp'].isna().sum())} "
            "unparsable/null values."
        )
    if df["Satellite_ID"].isna().any():
        raise ValueError("Satellite_ID contains null values.")
    df["Satellite_ID"] = df["Satellite_ID"].astype(str)
    df = df.sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)
    report = cadence_report(df)
    if report["duplicate_epochs"]:
        raise ValueError(
            "Dataset contains duplicate (Satellite_ID, Timestamp) rows: "
            f"{report['duplicate_epochs']}. Resolve the point-in-time join first."
        )
    return apply_target_validity_contract(df), report


def load_and_clean_data(
    data_path: str,
    outlier_threshold: float = OUTLIER_THRESHOLD_3D,
    train_end_date: str = TRAIN_END_DATE,
    strict_cadence: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load/validate rows and split chronologically without target filtering."""
    print("=" * 65)
    print("Loading & Validating GNSS Telemetry Data")
    print("=" * 65)
    df, report = _read_data_frame(data_path)
    if strict_cadence and report["irregular_steps"]:
        raise ValueError(
            "Dataset is not on an exact 15-minute cadence: "
            f"{report['irregular_by_satellite']}"
        )

    # Auto-adjust split_time to Day 1-7 Train, Day 8 Test if train_end_date is outside the dataset span
    min_t, max_t = df["Timestamp"].min(), df["Timestamp"].max()
    configured_split = pd.Timestamp(train_end_date)
    if configured_split < min_t or configured_split > max_t:
        split_time = max_t - pd.Timedelta(hours=23, minutes=45)
    else:
        split_time = configured_split

    train_df = df[df["Timestamp"] < split_time].copy()
    test_df = df[df["Timestamp"] >= split_time].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Split at {split_time} produced train={len(train_df)} and "
            f"test={len(test_df)} rows. Choose a boundary inside the data range."
        )

    # Compatibility name: this is now a train-known, not future-complete cohort.
    complete_sats = sorted(train_df["Satellite_ID"].unique().tolist())
    test_df = test_df[test_df["Satellite_ID"].isin(complete_sats)].copy()
    high_error_rows = 0
    if "3D_Orbit_Error" in df.columns:
        values = pd.to_numeric(df["3D_Orbit_Error"], errors="coerce")
        high_error_rows = int((values >= float(outlier_threshold)).fillna(False).sum())
    metadata = {
        "split_time": split_time.isoformat(),
        "cadence": report,
        "sp3_clock_sentinel_rows": int(df["SP3_Clock_Sentinel"].sum()),
        "high_orbit_error_rows_retained": high_error_rows,
    }
    train_df.attrs["data_contract"] = metadata
    test_df.attrs["data_contract"] = metadata
    gps = sum(sat.startswith("G") for sat in complete_sats)
    glo = sum(sat.startswith("R") for sat in complete_sats)
    geo = sum("GEO" in sat for sat in complete_sats)
    meo = sum("MEO" in sat for sat in complete_sats)
    type_str = f"GPS={gps}, GLONASS={glo}" if (gps or glo) else f"GEO={geo}, MEO={meo}"
    print(f"  Train-known satellites: {len(complete_sats)} ({type_str})")
    print(f"  Training records      : {len(train_df):,}")
    print(f"  Testing records       : {len(test_df):,}")
    print(f"  SP3 clock sentinels   : {metadata['sp3_clock_sentinel_rows']:,} (masked)")
    print(f"  Irregular cadence gaps: {report['irregular_steps']:,} (windows purged)")
    print(f"  Large targets retained: {high_error_rows:,}")
    return train_df, test_df, complete_sats


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cyclical time features and causal per-satellite rolling means."""
    result = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(result["Timestamp"]):
        result["Timestamp"] = pd.to_datetime(result["Timestamp"], errors="raise")
    result = result.sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)
    minutes = result["Timestamp"].dt.hour * 60 + result["Timestamp"].dt.minute
    result["time_sin"] = np.sin(2 * np.pi * minutes / 1440.0)
    result["time_cos"] = np.cos(2 * np.pi * minutes / 1440.0)
    for column in TARGET_COLS_4:
        if column in result.columns:
            result[f"{column}_roll_mean"] = result.groupby(
                "Satellite_ID", sort=False
            )[column].transform(lambda values: values.rolling(8, min_periods=1).mean())

    # Broadcast state is available at forecast issue time and provides a basis
    # for later ECEF->RIC residual modelling. Derivatives use only the current
    # and previous epoch; no centered/future difference is used.
    position_cols = ["Broadcast_X", "Broadcast_Y", "Broadcast_Z"]
    if set(position_cols).issubset(result.columns):
        elapsed = result.groupby("Satellite_ID", sort=False)["Timestamp"].diff().dt.total_seconds()
        for axis in ("X", "Y", "Z"):
            position = pd.to_numeric(result[f"Broadcast_{axis}"], errors="coerce")
            delta = position.groupby(result["Satellite_ID"], sort=False).diff()
            result[f"Broadcast_V{axis}"] = delta / elapsed
        coordinates = result[position_cols].apply(pd.to_numeric, errors="coerce")
        result["Broadcast_Radius"] = np.sqrt(np.square(coordinates).sum(axis=1))
        xy_radius = np.sqrt(coordinates["Broadcast_X"] ** 2 + coordinates["Broadcast_Y"] ** 2)
        result["Broadcast_Phase_Sin"] = coordinates["Broadcast_Y"] / xy_radius.replace(0, np.nan)
        result["Broadcast_Phase_Cos"] = coordinates["Broadcast_X"] / xy_radius.replace(0, np.nan)
    if "Broadcast_Clock" in result.columns:
        elapsed = result.groupby("Satellite_ID", sort=False)["Timestamp"].diff().dt.total_seconds()
        broadcast_clock = pd.to_numeric(result["Broadcast_Clock"], errors="coerce")
        result["Broadcast_Clock_Drift"] = (
            broadcast_clock.groupby(result["Satellite_ID"], sort=False).diff() / elapsed
        )
    return result


def _fit_standard_scaler(
    frame: pd.DataFrame,
    columns: Sequence[str],
    context: str,
) -> StandardScaler:
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    empty = [column for column in columns if not np.isfinite(values[column]).any()]
    if empty:
        raise ValueError(
            f"Cannot fit {context} scaler; no finite training observations for {empty}."
        )
    return StandardScaler().fit(values)


def _keras_validation_start(
    frame: pd.DataFrame,
    seq_len: int,
    horizon: int,
    val_split: float,
) -> pd.Timestamp:
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")
    timestamps = pd.DatetimeIndex(frame["Timestamp"].drop_duplicates().sort_values())
    minimum = seq_len + 2 * horizon
    if len(timestamps) < minimum:
        raise ValueError(
            "Insufficient chronological history for disjoint train/validation "
            f"labels: need {minimum} unique epochs, found {len(timestamps)}."
        )
    requested = max(horizon, int(np.ceil(len(timestamps) * val_split)))
    maximum = len(timestamps) - (seq_len + horizon)
    val_steps = min(requested, maximum)
    if val_steps < horizon:
        raise ValueError(f"Validation has {val_steps} epochs; horizon requires {horizon}.")
    return pd.Timestamp(timestamps[-val_steps])


def scale_datasets_keras(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: List[str] = TARGET_COLS_4,
    val_split: float = 0.1,
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_HORIZON,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale supported targets once, fitting only the chronological train fold."""
    columns = _validate_target_columns(target_cols)
    train, test = train_df.copy(), test_df.copy()
    val_start = _keras_validation_start(train, seq_len, horizon, val_split)
    scaler = _fit_standard_scaler(
        train[train["Timestamp"] < val_start], columns, "target"
    )
    train.loc[:, columns] = scaler.transform(train.loc[:, columns])
    test.loc[:, columns] = scaler.transform(test.loc[:, columns])
    train.attrs["scaler_fit_end_exclusive"] = val_start.isoformat()
    test.attrs["scaler_fit_end_exclusive"] = val_start.isoformat()
    return train, test, scaler


def _is_contiguous(timestamps: np.ndarray, interval: pd.Timedelta) -> bool:
    if len(timestamps) < 2:
        return True
    values = timestamps.astype("datetime64[ns]").astype(np.int64)
    return bool(np.all(np.diff(values) == interval.value))


def make_sliding_sequences(
    series: np.ndarray,
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_HORIZON,
    timestamps: Optional[Sequence] = None,
    interval: str | pd.Timedelta = EXPECTED_INTERVAL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct contiguous windows, dropping invalid-label windows for legacy use."""
    values = np.asarray(series)
    if values.ndim != 2:
        raise ValueError(f"series must have shape [time, features], got {values.shape}")
    required = seq_len + horizon
    if seq_len <= 0 or horizon <= 0 or len(values) < required:
        raise ValueError(
            f"Insufficient history: need at least {required} rows with positive "
            f"seq_len/horizon, found {len(values)}."
        )
    timestamp_values = None
    expected = _as_interval(interval)
    if timestamps is not None:
        timestamp_values = np.asarray(pd.to_datetime(timestamps), dtype="datetime64[ns]")
        if len(timestamp_values) != len(values):
            raise ValueError("timestamps and series must have the same length.")
    features, targets = [], []
    for start in range(len(values) - required + 1):
        stop = start + required
        if timestamp_values is not None and not _is_contiguous(
            timestamp_values[start:stop], expected
        ):
            continue
        target = values[start + seq_len : stop]
        if not np.isfinite(target).all():
            continue
        features.append(np.nan_to_num(values[start : start + seq_len], nan=0.0))
        targets.append(target)
    if not features:
        raise ValueError(
            "No valid sliding windows remain after cadence and label validation."
        )
    return np.asarray(features, np.float32), np.asarray(targets, np.float32)


def build_keras_sequences(
    train_df: pd.DataFrame,
    complete_sats: List[str],
    seq_len: int = SEQ_LEN,
    horizon: int = FORECAST_HORIZON,
    target_cols: List[str] = TARGET_COLS_4,
    val_split: float = 0.1,
    seed: int = DEFAULT_SEED,
    return_metadata: bool = False,
):
    """Build chronological, boundary-purged train/validation legacy sequences."""
    del seed
    columns = _validate_target_columns(target_cols)
    val_start = _keras_validation_start(train_df, seq_len, horizon, val_split)
    parts = {
        split: {key: [] for key in ("X", "Y", "input_ts", "label_ts", "sat")}
        for split in ("train", "val")
    }
    skipped_gap = skipped_invalid = purged = 0
    for satellite_id in complete_sats:
        sat_df = train_df[train_df["Satellite_ID"] == satellite_id].sort_values(
            "Timestamp"
        )
        values = sat_df[columns].to_numpy(dtype=np.float64)
        timestamps = sat_df["Timestamp"].to_numpy(dtype="datetime64[ns]")
        validity = np.isfinite(values)
        for index, column in enumerate(columns):
            valid_col = f"{column}{VALID_SUFFIX}"
            if valid_col in sat_df:
                validity[:, index] &= sat_df[valid_col].to_numpy(dtype=bool)
        for start in range(len(sat_df) - seq_len - horizon + 1):
            stop = start + seq_len + horizon
            window_ts = timestamps[start:stop]
            if not _is_contiguous(window_ts, EXPECTED_INTERVAL):
                skipped_gap += 1
                continue
            label_slice = slice(start + seq_len, stop)
            if not validity[label_slice].all():
                skipped_invalid += 1
                continue
            label_ts = timestamps[label_slice]
            if pd.Timestamp(label_ts[-1]) < val_start:
                split = "train"
            elif pd.Timestamp(label_ts[0]) >= val_start:
                split = "val"
            else:
                purged += 1
                continue
            parts[split]["X"].append(
                np.nan_to_num(values[start : start + seq_len], nan=0.0)
            )
            parts[split]["Y"].append(values[label_slice])
            parts[split]["input_ts"].append(timestamps[start : start + seq_len])
            parts[split]["label_ts"].append(label_ts)
            parts[split]["sat"].append(str(satellite_id))

    def stack(split, key, tail, dtype):
        items = parts[split][key]
        return np.asarray(items, dtype=dtype) if items else np.empty((0, *tail), dtype=dtype)

    n_features = len(columns)
    X_train = stack("train", "X", (seq_len, n_features), np.float32)
    y_train = stack("train", "Y", (horizon, n_features), np.float32)
    X_val = stack("val", "X", (seq_len, n_features), np.float32)
    y_val = stack("val", "Y", (horizon, n_features), np.float32)
    if not len(X_train) or not len(X_val):
        raise ValueError(
            "No usable chronological train/validation samples. Each satellite "
            f"needs at least {seq_len + 2 * horizon} contiguous epochs and valid labels."
        )
    print(f"  Training sequences   : {X_train.shape}")
    print(f"  Validation sequences : {X_val.shape}")
    print(
        f"  Purged windows        : boundary={purged}, gaps={skipped_gap}, "
        f"invalid_labels={skipped_invalid}"
    )
    if not return_metadata:
        return X_train, y_train, X_val, y_val
    metadata = {
        "validation_start": val_start.isoformat(),
        "train_input_timestamps": stack("train", "input_ts", (seq_len,), "datetime64[ns]"),
        "train_label_timestamps": stack("train", "label_ts", (horizon,), "datetime64[ns]"),
        "val_input_timestamps": stack("val", "input_ts", (seq_len,), "datetime64[ns]"),
        "val_label_timestamps": stack("val", "label_ts", (horizon,), "datetime64[ns]"),
        "train_satellite_ids": np.asarray(parts["train"]["sat"], dtype=object),
        "val_satellite_ids": np.asarray(parts["val"]["sat"], dtype=object),
        "purged_boundary_windows": purged,
        "skipped_noncontiguous_windows": skipped_gap,
        "skipped_invalid_target_windows": skipped_invalid,
    }
    return X_train, y_train, X_val, y_val, metadata


def _resolve_split_boundaries(
    df: pd.DataFrame,
    input_window: int,
    forecast_horizon: int,
    test_size: float,
    val_ratio: float,
    train_end_date: str,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not 0.0 < test_size < 1.0 or not 0.0 < val_ratio < 1.0:
        raise ValueError("test_size and val_ratio must each be in (0, 1).")
    timestamps = pd.DatetimeIndex(df["Timestamp"].drop_duplicates().sort_values())
    minimum = input_window + 3 * forecast_horizon
    if len(timestamps) < minimum:
        raise ValueError(
            "Insufficient history for purged train/validation/test blocks: "
            f"need {minimum} unique epochs, found {len(timestamps)}."
        )
    configured = pd.Timestamp(train_end_date)
    if timestamps[0] < configured <= timestamps[-1]:
        test_start = configured
    else:
        test_steps = max(
            forecast_horizon,
            int(np.ceil(len(timestamps) * test_size * val_ratio)),
        )
        test_start = pd.Timestamp(timestamps[-test_steps])
    pretest = timestamps[timestamps < test_start]
    test_epochs = timestamps[timestamps >= test_start]
    if len(test_epochs) < forecast_horizon:
        raise ValueError(
            f"Test block has {len(test_epochs)} epochs; horizon needs {forecast_horizon}."
        )
    requested_val = max(
        forecast_horizon,
        int(np.ceil(len(timestamps) * test_size * (1.0 - val_ratio))),
    )
    max_val = len(pretest) - (input_window + forecast_horizon)
    val_steps = min(requested_val, max_val)
    if val_steps < forecast_horizon:
        raise ValueError(
            "Training history before test is too short for disjoint train and "
            "validation forecast blocks."
        )
    return pd.Timestamp(pretest[-val_steps]), test_start


def _has_contiguous_training_window(
    sat_df: pd.DataFrame,
    validation_start: pd.Timestamp,
    input_window: int,
    forecast_horizon: int,
    interval: pd.Timedelta,
) -> bool:
    timestamps = sat_df.loc[
        sat_df["Timestamp"] < validation_start, "Timestamp"
    ].sort_values().to_numpy(dtype="datetime64[ns]")
    required = input_window + forecast_horizon
    return any(
        _is_contiguous(timestamps[start : start + required], interval)
        for start in range(len(timestamps) - required + 1)
    )


def prepare_pytorch_datasets(
    data_path: str,
    input_window: int = SEQ_LEN,
    forecast_horizon: int = FORECAST_HORIZON,
    spike_threshold: float = SPIKE_THRESHOLD,
    batch_size: int = 32,
    test_size: float = 0.3,
    val_ratio: float = 0.5,
    train_end_date: str = TRAIN_END_DATE,
    interval: str | pd.Timedelta = EXPECTED_INTERVAL,
    seed: int = DEFAULT_SEED,
    feature_cols: Optional[Sequence[str]] = None,
    include_physical_features: bool = True,
) -> Dict:
    """Build masked PyTorch data with chronological, disjoint label blocks.

    Loaders yield ``(features, targets, satellite, spikes, target_mask)``.
    """
    from torch.utils.data import DataLoader, Dataset

    if input_window <= 0 or forecast_horizon <= 0 or batch_size <= 0:
        raise ValueError("input_window, forecast_horizon, and batch_size must be positive.")
    expected = _as_interval(interval)

    class GNSSPyTorchDataset(Dataset):
        def __init__(self, X, Y, SAT, SPIKE, TARGET_MASK):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.Y = torch.as_tensor(Y, dtype=torch.float32)
            self.SAT = torch.as_tensor(SAT, dtype=torch.long)
            self.SPIKE = torch.as_tensor(SPIKE, dtype=torch.float32)
            self.TARGET_MASK = torch.as_tensor(TARGET_MASK, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return (
                self.X[index], self.Y[index], self.SAT[index],
                self.SPIKE[index], self.TARGET_MASK[index],
            )

    df, cadence = _read_data_frame(data_path)
    source_rows = int(len(df))
    source_sentinel_rows = int(df["SP3_Clock_Sentinel"].sum())
    df = engineer_features(df)
    target_cols = list(TARGET_COLS_4)
    if feature_cols is None:
        selected_feature_cols = list(FEATURE_COLS_PYTORCH)
        if include_physical_features:
            selected_feature_cols.extend(
                column for column in PHYSICAL_FEATURE_COLS if column in df.columns
            )
    else:
        selected_feature_cols = list(feature_cols)
    if len(selected_feature_cols) != len(set(selected_feature_cols)):
        raise ValueError("feature_cols contains duplicate columns.")
    missing = sorted(set(selected_feature_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Engineered dataset is missing feature columns: {missing}")
    missing_target_history = sorted(set(target_cols) - set(selected_feature_cols))
    if missing_target_history:
        raise ValueError(
            "feature_cols must retain all four historical targets for residual and "
            f"naive baselines; missing {missing_target_history}."
        )
    validation_start, test_start = _resolve_split_boundaries(
        df, input_window, forecast_horizon, test_size, val_ratio, train_end_date
    )

    # Vocabulary/cohort depends only on usable training history, never future labels.
    known_satellites = [
        str(satellite_id)
        for satellite_id, sat_df in df.groupby("Satellite_ID", sort=True)
        if _has_contiguous_training_window(
            sat_df, validation_start, input_window, forecast_horizon, expected
        )
    ]
    if not known_satellites:
        raise ValueError(
            f"No satellite has contiguous training history before {validation_start}."
        )
    df = df[df["Satellite_ID"].isin(known_satellites)].copy()
    sat_encoder = LabelEncoder().fit(known_satellites)
    df["sat_idx"] = sat_encoder.transform(df["Satellite_ID"])

    scaler_fit_df = df[df["Timestamp"] < validation_start]
    feature_scaler = _fit_standard_scaler(
        scaler_fit_df, selected_feature_cols, "feature"
    )
    target_scaler = _fit_standard_scaler(scaler_fit_df, target_cols, "target")
    # Separate arrays prevent the historical feature-then-target double transform.
    feature_values = feature_scaler.transform(df[selected_feature_cols]).astype(np.float32)
    feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=0.0, neginf=0.0)
    target_values = target_scaler.transform(df[target_cols]).astype(np.float32)
    row_mask = np.column_stack(
        [df[f"{column}{VALID_SUFFIX}"].to_numpy(bool) for column in target_cols]
    )
    row_mask &= np.isfinite(target_values)
    target_values = np.where(row_mask, target_values, 0.0).astype(np.float32)

    working = df[["Timestamp", "Satellite_ID", "sat_idx"]].copy()
    working["_row"] = np.arange(len(working), dtype=np.int64)
    keys = ("X", "Y", "SAT", "SPIKE", "MASK", "input_ts", "label_ts", "satellite_id")
    parts = {split: {key: [] for key in keys} for split in ("train", "val", "test")}
    purged_boundary = skipped_gap = skipped_all_invalid = 0
    required = input_window + forecast_horizon
    for satellite_id, sat_df in working.groupby("Satellite_ID", sort=True):
        sat_df = sat_df.sort_values("Timestamp")
        rows = sat_df["_row"].to_numpy(np.int64)
        timestamps = sat_df["Timestamp"].to_numpy(dtype="datetime64[ns]")
        for start in range(len(sat_df) - required + 1):
            stop = start + required
            if not _is_contiguous(timestamps[start:stop], expected):
                skipped_gap += 1
                continue
            input_rows = rows[start : start + input_window]
            label_rows = rows[start + input_window : stop]
            label_ts = timestamps[start + input_window : stop]
            label_mask = row_mask[label_rows]
            if not label_mask.any():
                skipped_all_invalid += 1
                continue
            label_start, label_end = pd.Timestamp(label_ts[0]), pd.Timestamp(label_ts[-1])
            if label_end < validation_start:
                split = "train"
            elif label_start >= validation_start and label_end < test_start:
                split = "val"
            elif label_start >= test_start:
                split = "test"
            else:
                purged_boundary += 1
                continue
            y_seq = target_values[label_rows]
            spike = (
                (np.abs(y_seq) > float(spike_threshold)) & label_mask
            ).any(axis=1).astype(np.float32)
            parts[split]["X"].append(feature_values[input_rows])
            parts[split]["Y"].append(y_seq)
            parts[split]["SAT"].append(int(sat_df["sat_idx"].iloc[0]))
            parts[split]["SPIKE"].append(spike)
            parts[split]["MASK"].append(label_mask.astype(np.float32))
            parts[split]["input_ts"].append(timestamps[start : start + input_window])
            parts[split]["label_ts"].append(label_ts)
            parts[split]["satellite_id"].append(str(satellite_id))

    def pack(split: str) -> Dict[str, np.ndarray]:
        if not parts[split]["X"]:
            raise ValueError(
                f"No {split} samples remain after cadence, target-mask, and "
                "chronological-boundary validation. Increase history or reduce windows."
            )
        return {
            "X": np.asarray(parts[split]["X"], np.float32),
            "Y": np.asarray(parts[split]["Y"], np.float32),
            "SAT": np.asarray(parts[split]["SAT"], np.int64),
            "SPIKE": np.asarray(parts[split]["SPIKE"], np.float32),
            "MASK": np.asarray(parts[split]["MASK"], np.float32),
            "input_ts": np.asarray(parts[split]["input_ts"], dtype="datetime64[ns]"),
            "label_ts": np.asarray(parts[split]["label_ts"], dtype="datetime64[ns]"),
            "satellite_id": np.asarray(parts[split]["satellite_id"], dtype=object),
        }

    packed = {split: pack(split) for split in ("train", "val", "test")}
    datasets = {
        split: GNSSPyTorchDataset(
            data["X"], data["Y"], data["SAT"], data["SPIKE"], data["MASK"]
        )
        for split, data in packed.items()
    }
    pin_memory = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(int(seed))
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            pin_memory=pin_memory, generator=generator,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False, pin_memory=pin_memory
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False, pin_memory=pin_memory
        ),
    }

    data_quality_report = {
        **cadence,
        "source_rows": source_rows,
        "sp3_clock_sentinel_rows": source_sentinel_rows,
        "rows_retained": int(len(df)),
        "rows_excluded_no_training_history": int(source_rows - len(df)),
        "skipped_noncontiguous_windows": skipped_gap,
        "skipped_all_invalid_target_windows": skipped_all_invalid,
        "physical_feature_columns": [
            column for column in PHYSICAL_FEATURE_COLS if column in selected_feature_cols
        ],
    }
    split_metadata = {
        "interval_minutes": float(expected / pd.Timedelta(minutes=1)),
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "scaler_fit_end_exclusive": validation_start.isoformat(),
        "purge_steps": int(forecast_horizon - 1),
        "purged_boundary_windows": purged_boundary,
        "sample_counts": {split: int(len(data["X"])) for split, data in packed.items()},
    }
    bundle: Dict[str, object] = {
        "train_loader": loaders["train"],
        "val_loader": loaders["val"],
        "test_loader": loaders["test"],
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "sat_encoder": sat_encoder,
        "satellite_classes": sat_encoder.classes_.tolist(),
        "num_features": len(selected_feature_cols),
        "num_satellites": len(sat_encoder.classes_),
        "output_dim": len(target_cols),
        "feature_cols": selected_feature_cols,
        "target_cols": target_cols,
        "target_feature_indices": [
            selected_feature_cols.index(column) for column in target_cols
        ],
        "split_metadata": split_metadata,
        "data_quality_report": data_quality_report,
    }
    for split, data in packed.items():
        bundle[f"X_{split}"] = data["X"]
        bundle[f"Y_{split}"] = data["Y"]
        bundle[f"SAT_{split}"] = data["SAT"]
        bundle[f"SPIKE_{split}"] = data["SPIKE"]
        bundle[f"TARGET_MASK_{split}"] = data["MASK"]
        bundle[f"INPUT_TIMESTAMPS_{split}"] = data["input_ts"]
        bundle[f"LABEL_TIMESTAMPS_{split}"] = data["label_ts"]
        bundle[f"SATELLITE_IDS_{split}"] = data["satellite_id"]
    return bundle


class FastGPUTensorLoader:
    """Direct tensor mini-batch loader for tensors already resident on device."""

    def __init__(
        self,
        tensors: tuple,
        batch_size: int = 128,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
    ):
        if not tensors:
            raise ValueError("At least one tensor is required.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        lengths = {len(tensor) for tensor in tensors}
        if len(lengths) != 1:
            raise ValueError(f"All tensors must have equal length, got {sorted(lengths)}")
        self.tensors = [tensor.to(device) if device is not None else tensor for tensor in tensors]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(self.tensors[0])
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self.device = device or self.tensors[0].device

    def __iter__(self):
        indices = (
            torch.randperm(self.num_samples, device=self.device)
            if self.shuffle
            else torch.arange(self.num_samples, device=self.device)
        )
        for index in range(self.num_batches):
            batch_idx = indices[index * self.batch_size : (index + 1) * self.batch_size]
            yield tuple(tensor[batch_idx] for tensor in self.tensors)

    def __len__(self):
        return self.num_batches

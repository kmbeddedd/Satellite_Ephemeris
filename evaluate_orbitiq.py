"""
ISRO SIH 2025 GNSS Orbit & Clock Error Model Evaluation
======================================================
Evaluates pre-trained deep LSTM networks and Random Forest baselines on the
ISRO SIH 2025 dataset (GEO and MEO orbits) for Problem Statement 25176.

Computes:
- MAE, RMSE, Max Error, 3D Euclidean Orbit Error
- Shapiro-Wilk Normality test on error residuals
- 8th-day forward forecasting values
- Structured JSON and Markdown summary metrics
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'x_error (m)',
    'y_error (m)',
    'z_error (m)',
    'satclockerror (m)',
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class KerasLSTMInference:
    """Zero-overhead forward inference engine for Keras Sequential LSTM models stored in HDF5."""

    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        self.layers_config = []
        self.weights = {}
        self._load_model()

    def _load_model(self):
        with h5py.File(self.h5_path, 'r') as f:
            cfg = json.loads(f.attrs['model_config'])
            layers = cfg['config']['layers']
            mw = f['model_weights']

            for l in layers:
                cls_name = l['class_name']
                if cls_name in ('LSTM', 'Dense'):
                    name = l['config']['name']
                    self.layers_config.append({
                        'name': name,
                        'class': cls_name,
                        'config': l['config']
                    })
                    # Extract datasets
                    ds = {}
                    def _collect(n, obj):
                        if isinstance(obj, h5py.Dataset):
                            ds[n.split('/')[-1]] = obj[:]
                    mw[name].visititems(_collect)
                    self.weights[name] = ds

    def _lstm_step(self, x_seq: np.ndarray, kernel: np.ndarray, recurrent_kernel: np.ndarray,
                   bias: np.ndarray, return_sequences: bool = True) -> np.ndarray:
        """Keras LSTM cell execution with [i, f, c, o] gate ordering."""
        if x_seq.ndim == 2:
            x_seq = x_seq[np.newaxis, ...]
        batch_size, seq_len, _ = x_seq.shape
        hidden_dim = recurrent_kernel.shape[0]

        h = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        c = np.zeros((batch_size, hidden_dim), dtype=np.float32)

        outputs = []
        for t in range(seq_len):
            xt = x_seq[:, t, :]
            gates = xt @ kernel + h @ recurrent_kernel + bias
            i_gate = sigmoid(gates[:, :hidden_dim])
            f_gate = sigmoid(gates[:, hidden_dim:2 * hidden_dim])
            c_cand = np.tanh(gates[:, 2 * hidden_dim:3 * hidden_dim])
            o_gate = sigmoid(gates[:, 3 * hidden_dim:])

            c = f_gate * c + i_gate * c_cand
            h = o_gate * np.tanh(c)
            outputs.append(h[:, np.newaxis, :])

        return np.concatenate(outputs, axis=1) if return_sequences else h

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run forward pass through the sequential network."""
        curr = x
        for layer in self.layers_config:
            name = layer['name']
            cls = layer['class']
            cfg = layer['config']
            w = self.weights[name]

            if cls == 'LSTM':
                ret_seq = cfg.get('return_sequences', False)
                curr = self._lstm_step(
                    curr,
                    w['kernel'],
                    w['recurrent_kernel'],
                    w['bias'],
                    return_sequences=ret_seq
                )
            elif cls == 'Dense':
                if curr.ndim == 3:
                    curr = curr[:, -1, :]
                act = cfg.get('activation', 'linear')
                curr = curr @ w['kernel'] + w['bias']
                if act == 'relu':
                    curr = np.maximum(0.0, curr)
                elif act == 'sigmoid':
                    curr = sigmoid(curr)
                elif act == 'tanh':
                    curr = np.tanh(curr)
        return curr


def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Standardize column names, interpolate missing data, cap outliers, and
    compute engineered features according to SIH 2025 PS 25176 requirements.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Rename common variations
    column_mapping = {
        'x_error  (m)': 'x_error (m)',
        'y_error  (m)': 'y_error (m)',
        'z_error  (m)': 'z_error (m)',
        'satclockerror  (m)': 'satclockerror (m)',
    }
    df.rename(columns=column_mapping, inplace=True)

    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df = df.sort_values('utc_time').reset_index(drop=True)

    # Missing value handling
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].interpolate(method='linear').bfill()

    # Outlier capping (IQR 3.0x threshold)
    for col in FEATURE_COLUMNS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        df[col] = df[col].clip(lower, upper)

    # Feature engineering (16 features expected by pre-trained scalers)
    df['total_position_error'] = np.sqrt(
        df['x_error (m)'] ** 2 +
        df['y_error (m)'] ** 2 +
        df['z_error (m)'] ** 2
    )
    df['hour'] = df['utc_time'].dt.hour
    df['day'] = df['utc_time'].dt.day
    df['day_of_week'] = df['utc_time'].dt.dayofweek

    for col in FEATURE_COLUMNS:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=3, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=3, min_periods=1).std().fillna(0.0)

    return df


def create_sequences(data: np.ndarray, target_idx: int, seq_length: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create input time sequences and corresponding target values."""
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


def evaluate_residual_distribution(residuals: np.ndarray) -> Dict[str, Any]:
    """
    Perform statistical analysis on prediction residuals, including the
    Shapiro-Wilk test for normality (key ISRO SIH 2025 requirement).
    """
    shapiro_stat, shapiro_p = stats.shapiro(residuals) if len(residuals) >= 3 else (1.0, 1.0)
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_err = float(np.max(np.abs(residuals)))
    std_err = float(np.std(residuals))

    return {
        'mae': mae,
        'rmse': rmse,
        'max_error': max_err,
        'std_error': std_err,
        'shapiro_w': float(shapiro_stat),
        'shapiro_p_value': float(shapiro_p),
        'is_normal': bool(shapiro_p > 0.05),
    }


def evaluate_orbitiq_dataset(
    data_path: str,
    orbit_type: str,
    models_dir: str,
    seq_length: int = 7,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Run evaluation for both Pretrained Deep LSTM models and Random Forest baseline.
    """
    df = preprocess_data(data_path)
    feature_cols = [c for c in df.columns if c != 'utc_time']

    # Load pre-trained scalers
    scalers_path = Path(models_dir) / 'scalers.pkl'
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    # Determine scaler prefix key: for MEO_Train2, fallback to MEO scalers
    lookup_orbit = 'GEO' if 'GEO' in orbit_type.upper() else 'MEO'

    results = {
        'dataset': Path(data_path).name,
        'orbit_type': orbit_type,
        'num_epochs': len(df),
        'date_start': str(df['utc_time'].min()),
        'date_end': str(df['utc_time'].max()),
        'lstm_pretrained': {},
        'random_forest': {},
        'day_8_forecast': {},
    }

    # 1. EVALUATE PRE-TRAINED LSTM MODELS
    test_predictions_lstm = {}
    test_ground_truth = {}

    for target in FEATURE_COLUMNS:
        target_idx = feature_cols.index(target)
        scaler_key = f'{lookup_orbit}_{target}'
        model_file = Path(models_dir) / f'{lookup_orbit}_{target}_model.h5'

        if scaler_key not in scalers or not model_file.exists():
            continue

        scaler = scalers[scaler_key]
        raw_data = df[feature_cols].values
        scaled_data = scaler.transform(raw_data)

        X, y_scaled = create_sequences(scaled_data, target_idx, seq_length)
        if len(X) < 5:
            continue

        # Split into train / test
        X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
            X, y_scaled, test_size=test_size, shuffle=False
        )

        # Run inference using zero-dependency Keras LSTM loader
        lstm = KerasLSTMInference(str(model_file))
        y_pred_scaled = lstm.predict(X_test).ravel()

        # Inverse transform predictions and ground truth
        dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
        dummy_pred[:, target_idx] = y_pred_scaled
        y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]

        dummy_true = np.zeros((len(y_test_scaled), len(feature_cols)))
        dummy_true[:, target_idx] = y_test_scaled
        y_true = scaler.inverse_transform(dummy_true)[:, target_idx]

        residuals = y_true - y_pred
        test_predictions_lstm[target] = y_pred
        test_ground_truth[target] = y_true

        metrics = evaluate_residual_distribution(residuals)
        results['lstm_pretrained'][target] = metrics

        # 8th day 1-step forecast from latest sequence
        last_seq = scaled_data[-seq_length:][np.newaxis, ...]
        pred_scaled_8th = lstm.predict(last_seq).ravel()[0]
        dummy_8th = np.zeros((1, len(feature_cols)))
        dummy_8th[0, target_idx] = pred_scaled_8th
        pred_8th = scaler.inverse_transform(dummy_8th)[0, target_idx]
        results['day_8_forecast'][f'LSTM_{target}'] = float(pred_8th)

    # Calculate 3D Orbit Error for LSTM if x, y, z are present
    if all(k in test_predictions_lstm for k in ['x_error (m)', 'y_error (m)', 'z_error (m)']):
        dx = test_ground_truth['x_error (m)'] - test_predictions_lstm['x_error (m)']
        dy = test_ground_truth['y_error (m)'] - test_predictions_lstm['y_error (m)']
        dz = test_ground_truth['z_error (m)'] - test_predictions_lstm['z_error (m)']
        e_3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        results['lstm_pretrained']['3d_position_error_mean_m'] = float(np.mean(e_3d))
        results['lstm_pretrained']['3d_position_error_max_m'] = float(np.max(e_3d))
        results['lstm_pretrained']['3d_position_error_rmse_m'] = float(np.sqrt(np.mean(e_3d ** 2)))

    # 2. EVALUATE RANDOM FOREST BASELINE
    test_predictions_rf = {}
    for target in FEATURE_COLUMNS:
        target_idx = feature_cols.index(target)
        scaler = StandardScaler()
        raw_data = df[feature_cols].values
        scaled_data = scaler.fit_transform(raw_data)

        X, y_scaled = create_sequences(scaled_data, target_idx, seq_length)
        if len(X) < 5:
            continue

        X_rf = X.reshape(X.shape[0], -1)
        X_train, X_test, y_train_scaled, y_test_scaled = train_test_split(
            X_rf, y_scaled, test_size=test_size, shuffle=False
        )

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train_scaled)
        y_pred_scaled = rf.predict(X_test)

        dummy_pred = np.zeros((len(y_pred_scaled), len(feature_cols)))
        dummy_pred[:, target_idx] = y_pred_scaled
        y_pred = scaler.inverse_transform(dummy_pred)[:, target_idx]

        dummy_true = np.zeros((len(y_test_scaled), len(feature_cols)))
        dummy_true[:, target_idx] = y_test_scaled
        y_true = scaler.inverse_transform(dummy_true)[:, target_idx]

        residuals = y_true - y_pred
        test_predictions_rf[target] = y_pred

        metrics = evaluate_residual_distribution(residuals)
        results['random_forest'][target] = metrics

        # 8th day forecast with RF
        last_seq = scaled_data[-seq_length:].reshape(1, -1)
        pred_scaled_8th = rf.predict(last_seq)[0]
        dummy_8th = np.zeros((1, len(feature_cols)))
        dummy_8th[0, target_idx] = pred_scaled_8th
        pred_8th = scaler.inverse_transform(dummy_8th)[0, target_idx]
        results['day_8_forecast'][f'RF_{target}'] = float(pred_8th)

    if all(k in test_predictions_rf for k in ['x_error (m)', 'y_error (m)', 'z_error (m)']):
        dx = test_ground_truth['x_error (m)'] - test_predictions_rf['x_error (m)']
        dy = test_ground_truth['y_error (m)'] - test_predictions_rf['y_error (m)']
        dz = test_ground_truth['z_error (m)'] - test_predictions_rf['z_error (m)']
        e_3d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        results['random_forest']['3d_position_error_mean_m'] = float(np.mean(e_3d))
        results['random_forest']['3d_position_error_max_m'] = float(np.max(e_3d))
        results['random_forest']['3d_position_error_rmse_m'] = float(np.sqrt(np.mean(e_3d ** 2)))

    return results


def print_formatted_summary(all_results: List[Dict[str, Any]]):
    """Print clean formatted tables for terminal inspection."""
    print("\n" + "=" * 90)
    print("ISRO SIH 2025 GNSS ERROR MODELLING - EVALUATION BENCHMARK RESULTS")
    print("=" * 90)

    for res in all_results:
        dataset = res['dataset']
        orbit = res['orbit_type']
        n_epochs = res['num_epochs']
        print(f"\nDataset: {dataset} ({orbit} Orbit, {n_epochs} Telemetry Epochs)")
        print(f"Span: {res['date_start']}  -->  {res['date_end']}")
        print("-" * 90)
        print(f"{'Target Residual':<22} | {'Model':<12} | {'MAE (m)':<9} | {'RMSE (m)':<9} | {'Max (m)':<9} | {'Shapiro-W':<9} | {'Normal?':<7}")
        print("-" * 90)

        for target in FEATURE_COLUMNS:
            # Pretrained LSTM
            if target in res['lstm_pretrained']:
                m = res['lstm_pretrained'][target]
                norm_str = "Yes" if m['is_normal'] else "No"
                print(f"{target:<22} | {'LSTM Pretr.':<12} | {m['mae']:<9.4f} | {m['rmse']:<9.4f} | {m['max_error']:<9.4f} | {m['shapiro_w']:<9.4f} | {norm_str:<7}")

            # Random Forest
            if target in res['random_forest']:
                m = res['random_forest'][target]
                norm_str = "Yes" if m['is_normal'] else "No"
                print(f"{'':<22} | {'RandomForest':<12} | {m['mae']:<9.4f} | {m['rmse']:<9.4f} | {m['max_error']:<9.4f} | {m['shapiro_w']:<9.4f} | {norm_str:<7}")

        print("-" * 90)
        if '3d_position_error_mean_m' in res['lstm_pretrained']:
            l3d = res['lstm_pretrained']
            print(f"Pretrained LSTM 3D Position Error -> Mean: {l3d['3d_position_error_mean_m']:.4f} m | RMSE: {l3d['3d_position_error_rmse_m']:.4f} m | Max: {l3d['3d_position_error_max_m']:.4f} m")
        if '3d_position_error_mean_m' in res['random_forest']:
            r3d = res['random_forest']
            print(f"Random Forest   3D Position Error -> Mean: {r3d['3d_position_error_mean_m']:.4f} m | RMSE: {r3d['3d_position_error_rmse_m']:.4f} m | Max: {r3d['3d_position_error_max_m']:.4f} m")

        print("\n8th-Day Single Step Forecast:")
        for k, v in res['day_8_forecast'].items():
            print(f"   * {k:<30}: {v:>10.6f} m")


def generate_markdown_report(all_results: List[Dict[str, Any]], output_path: str):
    """Generate a clean Markdown summary report of the evaluation."""
    lines = [
        "# ISRO SIH 2025 GNSS Orbit & Clock Error Modeling Evaluation Report",
        "",
        "Evaluation results for pre-trained LSTM neural networks and Random Forest models on ISRO SIH 2025 dataset.",
        "",
    ]
    for res in all_results:
        dataset = res['dataset']
        orbit = res['orbit_type']
        n_epochs = res['num_epochs']
        lines.append(f"## Dataset: `{dataset}` ({orbit} Orbit, {n_epochs} Epochs)")
        lines.append(f"- **Time Span**: `{res['date_start']}` to `{res['date_end']}`")
        lines.append("")
        lines.append("| Target Residual | Model | MAE (m) | RMSE (m) | Max Error (m) | Shapiro-Wilk W | p-value | Normal? |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for target in FEATURE_COLUMNS:
            if target in res['lstm_pretrained']:
                m = res['lstm_pretrained'][target]
                norm = "Yes" if m['is_normal'] else "No"
                lines.append(f"| `{target}` | **Pretrained LSTM** | {m['mae']:.4f} | {m['rmse']:.4f} | {m['max_error']:.4f} | {m['shapiro_w']:.4f} | {m['shapiro_p_value']:.4e} | {norm} |")
            if target in res['random_forest']:
                m = res['random_forest'][target]
                norm = "Yes" if m['is_normal'] else "No"
                lines.append(f"| `{target}` | Random Forest | {m['mae']:.4f} | {m['rmse']:.4f} | {m['max_error']:.4f} | {m['shapiro_w']:.4f} | {m['shapiro_p_value']:.4e} | {norm} |")
        lines.append("")
        if '3d_position_error_mean_m' in res['lstm_pretrained']:
            l3d = res['lstm_pretrained']
            lines.append(f"- **LSTM 3D Position Error**: Mean = `{l3d['3d_position_error_mean_m']:.4f} m`, RMSE = `{l3d['3d_position_error_rmse_m']:.4f} m`, Max = `{l3d['3d_position_error_max_m']:.4f} m`")
        if '3d_position_error_mean_m' in res['random_forest']:
            r3d = res['random_forest']
            lines.append(f"- **RF 3D Position Error**: Mean = `{r3d['3d_position_error_mean_m']:.4f} m`, RMSE = `{r3d['3d_position_error_rmse_m']:.4f} m`, Max = `{r3d['3d_position_error_max_m']:.4f} m`")
        lines.append("")
        lines.append("### 8th-Day Forward Predictions:")
        for k, v in res['day_8_forecast'].items():
            lines.append(f"- `{k}`: `{v:.6f} m`")
        lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Evaluate OrbitIQ ISRO SIH 2025 GNSS Error Models")
    parser.add_argument('--data-dir', type=str, default='data/orbitiq', help='Directory containing OrbitIQ datasets')
    parser.add_argument('--models-dir', type=str, default='models/orbitiq_pretrained', help='Directory containing pretrained .h5 models and scalers')
    parser.add_argument('--output-dir', type=str, default='orbitiq_results', help='Directory to store evaluation reports')
    parser.add_argument('--all', action='store_true', help='Evaluate all available datasets (GEO, MEO, MEO2)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = [
        (data_dir / 'DATA_GEO_Train.csv', 'GEO'),
        (data_dir / 'DATA_MEO_Train.csv', 'MEO'),
        (data_dir / 'DATA_MEO_Train2.csv', 'MEO'),
    ]

    all_results = []
    for csv_file, orbit_type in datasets_to_run:
        if csv_file.exists():
            print(f"Evaluating {csv_file.name} ({orbit_type})...")
            res = evaluate_orbitiq_dataset(
                str(csv_file),
                orbit_type=orbit_type,
                models_dir=str(models_dir)
            )
            all_results.append(res)
        else:
            print(f"Warning: Dataset file {csv_file} not found.")

    # Print summary
    print_formatted_summary(all_results)

    # Save to JSON & Markdown
    report_json = output_dir / 'evaluation_metrics.json'
    with open(report_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    report_md = output_dir / 'EVALUATION_REPORT.md'
    generate_markdown_report(all_results, str(report_md))

    print(f"\nEvaluation reports successfully exported to:\n - {report_json}\n - {report_md}")


if __name__ == '__main__':
    main()

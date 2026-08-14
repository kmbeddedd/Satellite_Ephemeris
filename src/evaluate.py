"""
Evaluation Metrics Computation (Multi-Horizon MAE/RMSE, Per-Satellite Breakdown)
"""

import json
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple
from src.config import HORIZON_MAP, TARGET_COLS_5


def compute_aggregate_metrics(
    all_actuals: Dict[str, np.ndarray],
    all_preds: Dict[str, np.ndarray],
    target_cols: List[str] = TARGET_COLS_5
) -> Tuple[Dict, Dict]:
    """
    Computes per-satellite and aggregate MAE/RMSE across all evaluated satellites.
    """
    per_sat_results = {}
    for sat_id in all_preds:
        metrics = {}
        for i, col in enumerate(target_cols):
            act = all_actuals[sat_id][:, i]
            pred = all_preds[sat_id][:, i]
            metrics[col] = {
                "MAE": float(mean_absolute_error(act, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(act, pred)))
            }
        per_sat_results[sat_id] = metrics

    # Aggregate across satellites
    aggregate = {
        col: {
            "Mean_MAE": float(np.mean([per_sat_results[s][col]["MAE"] for s in per_sat_results])),
            "Mean_RMSE": float(np.mean([per_sat_results[s][col]["RMSE"] for s in per_sat_results]))
        }
        for col in target_cols
    }

    return per_sat_results, aggregate


def compute_multi_horizon_metrics(
    all_actuals: Dict[str, np.ndarray],
    all_preds: Dict[str, np.ndarray],
    target_cols: List[str] = TARGET_COLS_5,
    horizons: Dict[str, int] = HORIZON_MAP
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates forecasting accuracy at specific operational validity windows
    (e.g., 15m, 30m, 1h, 2h, 6h, 12h, 24h).
    """
    horizon_results = {}
    for label, steps in horizons.items():
        h_mae = {col: [] for col in target_cols}
        for sat_id in all_preds:
            for i, col in enumerate(target_cols):
                act_slice = all_actuals[sat_id][:steps, i]
                pred_slice = all_preds[sat_id][:steps, i]
                h_mae[col].append(mean_absolute_error(act_slice, pred_slice))

        horizon_results[label] = {
            col: float(np.mean(h_mae[col])) for col in target_cols
        }

    return horizon_results


def compute_tensor_horizon_metrics(
    actual_real: np.ndarray,
    pred_real: np.ndarray,
    target_cols: List[str],
    horizons: Dict[str, int] = HORIZON_MAP
) -> Dict[str, Dict[str, float]]:
    """
    Computes multi-horizon metrics from unified 3D numpy arrays (N, Horizon, Features).
    """
    horizon_metrics = {}
    for label, step in horizons.items():
        idx = min(step - 1, actual_real.shape[1] - 1)
        pred_slice = pred_real[:, idx, :]
        target_slice = actual_real[:, idx, :]

        metrics = {}
        for c, col in enumerate(target_cols):
            metrics[f"{col}_MAE"] = float(np.mean(np.abs(pred_slice[:, c] - target_slice[:, c])))
            metrics[f"{col}_RMSE"] = float(np.sqrt(np.mean((pred_slice[:, c] - target_slice[:, c]) ** 2)))

        metrics["Overall_MAE"] = float(np.mean(np.abs(pred_slice - target_slice)))
        metrics["Overall_RMSE"] = float(np.sqrt(np.mean((pred_slice - target_slice) ** 2)))
        horizon_metrics[label] = metrics

    return horizon_metrics


def print_metrics_summary(
    aggregate: Dict,
    horizon_results: Dict,
    target_cols: List[str] = TARGET_COLS_5
) -> None:
    """
    Prints human-readable tabular metrics summary.
    """
    print("\n" + "=" * 65)
    print("EVALUATION METRICS SUMMARY (24-Hour Prediction)")
    print("=" * 65)
    print(f"  {'Target':<22}  {'Mean MAE':>14}  {'Mean RMSE':>14}")
    print("  " + "-" * 54)
    for col in target_cols:
        print(f"  {col:<22}  {aggregate[col]['Mean_MAE']:>14.6f}  {aggregate[col]['Mean_RMSE']:>14.6f}")

    print("\n" + "=" * 80)
    print("MULTI-HORIZON MAE BREAKDOWN")
    print("=" * 80)
    print(f"  {'Horizon':<12}", end="")
    for col in target_cols:
        short = col.replace("Error_", "").replace("3D_Orbit_", "3D_")
        print(f"  {short:>12}", end="")
    print()
    print("  " + "-" * 80)

    for label, mets in horizon_results.items():
        print(f"  {label:<12}", end="")
        for col in target_cols:
            val = mets.get(col, mets.get(f"{col}_MAE", 0.0))
            print(f"  {val:>12.6f}", end="")
        print()


def save_metrics_summary(
    filepath: str,
    model_name: str,
    aggregate: Dict,
    horizon_results: Dict,
    per_sat_results: Optional[Dict] = None,
    extra_meta: Optional[Dict] = None
) -> None:
    """
    Saves comprehensive metrics summary to JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    summary = {
        "model": model_name,
        "aggregate_metrics": aggregate,
        "multi_horizon_metrics": horizon_results
    }
    if per_sat_results is not None:
        summary["per_satellite_metrics"] = per_sat_results
    if extra_meta is not None:
        summary["metadata"] = extra_meta

    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Metrics saved to: {filepath}")

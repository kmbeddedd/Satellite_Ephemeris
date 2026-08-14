"""
Publication-Quality Visualization and Diagnostic Plotting Functions
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script execution
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from typing import Dict, List
from src.config import TARGET_COLS_5, TARGET_COLS_4, HORIZON_MAP


def plot_training_history(history_dict: dict, save_path: str, title: str = "Training History") -> None:
    """
    Plots training and validation loss & metrics progression over epochs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Loss
    if "loss" in history_dict and "val_loss" in history_dict:
        axes[0].plot(history_dict["loss"], label="Train Loss", color="#1565C0", lw=1.8)
        axes[0].plot(history_dict["val_loss"], label="Val Loss", color="#E53935", lw=1.8)
        axes[0].set_title("Loss Convergence")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # MAE or secondary metric
    metric_key = "mae" if "mae" in history_dict else ("val_mae" if "val_mae" in history_dict else None)
    if metric_key and "val_mae" in history_dict:
        axes[1].plot(history_dict.get("mae", []), label="Train MAE", color="#1565C0", lw=1.8)
        axes[1].plot(history_dict["val_mae"], label="Val MAE", color="#E53935", lw=1.8)
        axes[1].set_title("Mean Absolute Error")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    elif "sigma_mean" in history_dict:
        axes[1].plot(history_dict["sigma_mean"], label="Mean Sigma", color="#2E7D32", lw=1.8)
        axes[1].plot(history_dict.get("spike_mean", []), label="Mean Spike Prob", color="#FF8F00", lw=1.8)
        axes[1].set_title("Uncertainty & Spike Parameters")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_prediction_vs_actual(
    all_actuals: Dict[str, np.ndarray],
    all_preds: Dict[str, np.ndarray],
    sat_ids: List[str],
    target_cols: List[str],
    save_path: str,
    title_prefix: str = "Day-8 Prediction vs Actual"
) -> None:
    """
    Overlays ground-truth telemetry with direct multi-step forecasted trajectories.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if not sat_ids:
        return

    fig, axes = plt.subplots(len(sat_ids), 2, figsize=(16, 4 * len(sat_ids)))
    if len(sat_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    fig.suptitle(f"{title_prefix} — Sample Satellites", fontsize=13, fontweight="bold")

    orbit_col = "3D_Orbit_Error" if "3D_Orbit_Error" in target_cols else "Error_X"
    clock_col = "Error_Clock"

    for row, sat in enumerate(sat_ids):
        for col_idx, (col_name, ylabel) in enumerate([
            (orbit_col, "3D Orbit Error (m)" if orbit_col == "3D_Orbit_Error" else "Error X (m)"),
            (clock_col, "Clock Error (s)")
        ]):
            ax = axes[row][col_idx]
            ci = target_cols.index(col_name)
            act = all_actuals[sat][:, ci]
            prd = all_preds[sat][:, ci]
            mae = float(np.mean(np.abs(act - prd)))

            ax.plot(act, color="#1565C0", lw=1.5, label="Actual Ground Truth")
            ax.plot(prd, color="#E53935", lw=1.5, ls="--", label="Forecasted Model")
            ax.set_title(f"{sat} — {ylabel}", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.annotate(
                f"MAE = {mae:.4f}",
                xy=(0.02, 0.88),
                xycoords="axes fraction",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            )

    for col_idx in range(2):
        axes[-1][col_idx].set_xlabel("Time step (15-min increments)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_multihorizon_heatmap(
    horizon_results: Dict[str, Dict],
    target_cols: List[str],
    save_path: str
) -> None:
    """
    Generates matrix heatmap of MAE progression across forecast horizons.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    h_labels = list(horizon_results.keys())
    col_labels = [c.replace("Error_", "").replace("3D_Orbit_", "3D_") for c in target_cols]

    mat = np.array([
        [horizon_results[h].get(col, horizon_results[h].get(f"{col}_MAE", 0.0)) for col in target_cols]
        for h in h_labels
    ])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(target_cols)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(h_labels)))
    ax.set_yticklabels(h_labels, fontsize=11)
    ax.set_title("Multi-Horizon MAE Heatmap (Mean Across Satellites)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="MAE")

    for r in range(len(h_labels)):
        for c in range(len(target_cols)):
            ax.text(
                c, r, f"{mat[r, c]:.4f}",
                ha="center", va="center", fontsize=8,
                color="black" if mat[r, c] < mat.max() * 0.6 else "white"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_residual_distributions(
    all_actuals: Dict[str, np.ndarray],
    all_preds: Dict[str, np.ndarray],
    target_cols: List[str],
    save_path: str
) -> None:
    """
    Generates residual histograms and fitted Gaussian densities to verify unbiased forecasts.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    col_labels = [c.replace("Error_", "").replace("3D_Orbit_", "3D_") for c in target_cols]
    fig, axes = plt.subplots(1, len(target_cols), figsize=(4.5 * len(target_cols), 4))
    if len(target_cols) == 1:
        axes = [axes]

    fig.suptitle("Prediction Residuals — Normality Check (Pooled Across Satellites)", fontsize=13, fontweight="bold")

    for i, (ax, col) in enumerate(zip(axes, target_cols)):
        resids = np.concatenate([
            all_actuals[s][:, i] - all_preds[s][:, i]
            for s in all_preds
        ])
        mu, sig = resids.mean(), resids.std()
        ax.hist(resids, bins=60, color="#5C6BC0", edgecolor="white", alpha=0.82, density=True)
        if sig > 1e-8:
            x = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
            ax.plot(
                x,
                np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
                color="#E53935", lw=2,
                label=f"N(μ={mu:.2e}, σ={sig:.2e})"
            )
        ax.set_title(col_labels[i], fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlabel("Residual Error")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_per_satellite_mae(
    all_results: Dict[str, Dict],
    target_cols: List[str],
    save_path: str
) -> None:
    """
    Bar chart comparing individual satellite prediction MAE for orbit and clock.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    sat_ids = sorted(all_results.keys())
    orbit_col = "3D_Orbit_Error" if "3D_Orbit_Error" in target_cols else "Error_X"
    clock_col = "Error_Clock"

    orbit_maes = [all_results[s][orbit_col]["MAE"] for s in sat_ids]
    clock_maes = [all_results[s][clock_col]["MAE"] for s in sat_ids]
    bar_colors = ["#1565C0" if s.startswith("G") else "#AD1457" for s in sat_ids]

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle("Per-Satellite MAE — Day-8 Evaluation", fontsize=13, fontweight="bold")

    for ax, vals, title, unit in [
        (axes[0], orbit_maes, f"{orbit_col.replace('_', ' ')} MAE (m)", "m"),
        (axes[1], clock_maes, "Clock Error MAE (s)", "s")
    ]:
        ax.bar(sat_ids, vals, color=bar_colors, edgecolor="white", lw=0.5)
        mean_val = np.mean(vals)
        ax.axhline(mean_val, color="black", lw=1.5, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(sat_ids)))
        ax.set_xticklabels(sat_ids, rotation=90, fontsize=7)
        ax.set_ylabel(f"MAE ({unit})")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(handles=[
            Patch(color="#1565C0", label="GPS (G)"),
            Patch(color="#AD1457", label="GLONASS (R)"),
            plt.Line2D([0], [0], color="black", ls="--", label=f"Mean = {mean_val:.4f} {unit}")
        ], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_probabilistic_uncertainty(
    actual: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    save_path: str,
    feature_idx: int = 0,
    feature_name: str = "Error_X"
) -> None:
    """
    Plots predictive mean forecast along with 1-sigma uncertainty confidence interval bands.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(14, 5))

    plt.plot(actual[:, feature_idx], label="Actual Ground Truth", color="black", lw=2)
    plt.plot(mu[:, feature_idx], label="Predicted Mean (μ)", color="#1565C0", lw=2)

    upper = mu[:, feature_idx] + sigma[:, feature_idx]
    lower = mu[:, feature_idx] - sigma[:, feature_idx]

    plt.fill_between(
        np.arange(len(upper)),
        lower,
        upper,
        color="#1565C0",
        alpha=0.25,
        label="±1σ Predictive Uncertainty"
    )

    plt.title(f"Probabilistic GNSS Forecasting with Uncertainty Bounds ({feature_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Forecast Timestep (15-min)")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_frequency_spectrum(
    actual: np.ndarray,
    predicted: np.ndarray,
    save_path: str,
    feature_idx: int = 0,
    feature_name: str = "Error_X"
) -> None:
    """
    Compares FFT spectral power density of ground truth vs predicted forecasts.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    act_fft = np.abs(np.fft.rfft(actual[:, feature_idx]))
    pred_fft = np.abs(np.fft.rfft(predicted[:, feature_idx]))

    plt.figure(figsize=(12, 4.5))
    plt.plot(act_fft, label="Actual Spectrum", color="black", lw=2)
    plt.plot(pred_fft, label="Predicted Spectrum", color="#E53935", lw=2, ls="--")
    plt.title(f"FFT Frequency Spectrum Consistency ({feature_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_diffusion_samples(
    actual: np.ndarray,
    samples: List[np.ndarray],
    save_path: str,
    feature_idx: int = 0,
    feature_name: str = "Error_X"
) -> None:
    """
    Visualizes stochastic multi-sample trajectory rollouts from the diffusion model.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.figure(figsize=(15, 6))

    plt.plot(actual[:, feature_idx], color="black", lw=2.5, label="Actual Ground Truth")
    for idx, s in enumerate(samples):
        plt.plot(s[:, feature_idx], color="#00897B", alpha=0.35, label="Diffusion Sample" if idx == 0 else "")

    plt.title(f"Multi-Sample Stochastic Diffusion Generation ({feature_name})", fontsize=12, fontweight="bold")
    plt.xlabel("Forecast Timestep (15-min)")
    plt.ylabel("Residual Error")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")

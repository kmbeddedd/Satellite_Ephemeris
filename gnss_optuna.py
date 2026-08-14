"""
╔══════════════════════════════════════════════════════════════════╗
║   GNSS Satellite Error Forecasting — BiLSTM + GRU Model         ║
║   Problem : Predict Day-8 clock & ephemeris errors               ║
║   Train   : Day 1–7  |  Test: Day 8  (15-min intervals)         ║
║   Targets : Error_X, Error_Y, Error_Z, 3D_Orbit_Error,          ║
║             Error_Clock                                          ║
║   Strategy: Shared model trained across ALL complete satellites  ║
╚══════════════════════════════════════════════════════════════════╝

Architecture:
    Input (96 steps × 5 features)
    → Bidirectional LSTM (64 units, return_sequences=True)
    → Dropout(0.2)
    → GRU (32 units)
    → Dropout(0.2)
    → LayerNormalization
    → Dense(64, ReLU)
    → Dense(96 × 5)  →  Reshape(96, 5)

Usage:
    python gnss_forecast.py --data path/to/FINAL_Data__1_.csv --output ./results

Requirements:
    pip install tensorflow scikit-learn pandas numpy matplotlib
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────
import os
import warnings
import argparse
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # Suppress TF info/warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                            # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, GRU, Dense, Dropout,
    Bidirectional, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ──────────────────────────────────────────────────────────────────
# CONFIG  — change these if needed
# ──────────────────────────────────────────────────────────────────
DATA_PATH        = "FINAL_Data (1).csv"   # override via --data
OUTPUT_DIR       = "./gnss_results"       # override via --output
TRAIN_END_DATE   = "2026-01-08 00:00:00"  # Day 8 starts here
SEQ_LEN          = 96                     # look-back  = 1 day (96 × 15 min)
FORECAST_HORIZON = 96                     # predict    = 1 day ahead
TARGET_COLS = [
    "Error_X", "Error_Y", "Error_Z",
    "3D_Orbit_Error", "Error_Clock"
]
OUTLIER_THRESHOLD = 50_000               # 3D_Orbit_Error cap (metres)
EPOCHS            = 60
BATCH_SIZE        = 64
SEED              = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="GNSS Error Forecasting")
    parser.add_argument("--data",   default=DATA_PATH,   help="Path to CSV file")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output directory for results")
    return parser.parse_args()

# ──────────────────────────────────────────────────────────────────
# STEP 1 — LOAD & CLEAN
# ──────────────────────────────────────────────────────────────────
def load_and_clean(data_path):
    print("=" * 65)
    print("STEP 1 — Loading & Cleaning Data")
    print("=" * 65)

    df = pd.read_csv(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)

    # ── Keep only satellites with a full 768-record series ──
    # (768 = 8 days × 96 timestamps/day)
    # Incomplete satellites (G20, R11, R28) are excluded — they would
    # introduce NaN gaps that break the sliding-window sequencer.
    sat_counts    = df.groupby("Satellite_ID").size()
    complete_sats = sorted(sat_counts[sat_counts == 768].index.tolist())
    df = df[df["Satellite_ID"].isin(complete_sats)].copy()

    # ── Remove gross orbit-error outliers ──
    # 15 records have 3D errors > 100 km (physically impossible).
    # Most are clustered at Day-1 01:00 — likely an initialisation artefact.
    # We cap at 50 km to avoid them dominating loss scaling.
    df = df[df["3D_Orbit_Error"] < OUTLIER_THRESHOLD].copy()

    # ── Train / Test split ──
    train_df = df[df["Timestamp"] <  pd.Timestamp(TRAIN_END_DATE)].copy()
    test_df  = df[df["Timestamp"] >= pd.Timestamp(TRAIN_END_DATE)].copy()

    # Constellation breakdown
    gps_sats  = [s for s in complete_sats if s.startswith("G")]
    glo_sats  = [s for s in complete_sats if s.startswith("R")]

    print(f"  Complete satellites : {len(complete_sats)}  "
          f"(GPS={len(gps_sats)}, GLONASS={len(glo_sats)})")
    print(f"  Training rows       : {len(train_df):,}  (Day 1–7)")
    print(f"  Test rows (Day 8)   : {len(test_df):,}")
    print(f"  Targets             : {TARGET_COLS}")

    return train_df, test_df, complete_sats

# ──────────────────────────────────────────────────────────────────
# STEP 2 — SCALE
# ──────────────────────────────────────────────────────────────────
def scale_data(train_df, test_df):
    """
    Global StandardScaler fitted on training data only.
    Scaling all 5 targets together keeps their relative magnitudes
    consistent across the shared model.
    """
    print("\nSTEP 2 — Scaling (StandardScaler, fit on train only)")

    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df  = test_df.copy()
    train_df[TARGET_COLS] = scaler.fit_transform(train_df[TARGET_COLS])
    test_df[TARGET_COLS]  = scaler.transform(test_df[TARGET_COLS])

    print(f"  Mean (train): {dict(zip(TARGET_COLS, scaler.mean_.round(6)))}")
    return train_df, test_df, scaler

# ──────────────────────────────────────────────────────────────────
# STEP 3 — SEQUENCE BUILDER
# ──────────────────────────────────────────────────────────────────
def make_sequences(series, seq_len, horizon):
    """
    Sliding-window sequence builder.
    series  : np.array  (T, F)
    Returns : X (N, seq_len, F),  y (N, horizon, F)
    """
    X, y = [], []
    for i in range(len(series) - seq_len - horizon + 1):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len : i + seq_len + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_all_sequences(train_df, complete_sats):
    print("\nSTEP 3 — Building sliding-window sequences")
    print(f"  Look-back = {SEQ_LEN} steps (1 day)  |  "
          f"Forecast = {FORECAST_HORIZON} steps (1 day)")

    X_all, y_all = [], []
    for sat_id in complete_sats:
        sat_data = train_df[train_df["Satellite_ID"] == sat_id][TARGET_COLS].values
        X_s, y_s = make_sequences(sat_data, SEQ_LEN, FORECAST_HORIZON)
        X_all.append(X_s)
        y_all.append(y_s)

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Shuffle so validation set spans all satellites
    idx   = np.random.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    split = int(0.9 * len(X_all))
    X_tr, X_val = X_all[:split], X_all[split:]
    y_tr, y_val = y_all[:split], y_all[split:]

    print(f"  Train sequences : {X_tr.shape}")
    print(f"  Val   sequences : {X_val.shape}")
    return X_tr, y_tr, X_val, y_val

# ──────────────────────────────────────────────────────────────────
# STEP 4 — MODEL
# ──────────────────────────────────────────────────────────────────
def build_model():
    """
    Bidirectional LSTM + GRU forecaster.

    Why this architecture:
    - BiLSTM captures forward AND backward temporal dependencies in the
      look-back window, useful because GNSS errors show both gradual drift
      and periodic patterns.
    - GRU (after BiLSTM) compresses the sequence into a single context
      vector efficiently — fewer parameters than a second LSTM.
    - LayerNorm stabilises training when error magnitudes vary wildly
      across satellites and constellations.
    - Huber loss is robust to the residual extreme outliers that survive
      our 50 km cap.
    - Direct multi-step output (all 96 future steps predicted at once)
      avoids compounding errors from autoregressive roll-out.
    """
    N_FEATURES = len(TARGET_COLS)

    inp = Input(shape=(SEQ_LEN, N_FEATURES), name="input")

    # Encoder — bidirectional LSTM
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm")(inp)
    x = Dropout(0.2, name="drop1")(x)

    # Encoder — GRU compression
    x = GRU(32, return_sequences=False, name="gru")(x)
    x = Dropout(0.2, name="drop2")(x)
    x = LayerNormalization(name="layernorm")(x)

    # Projection head
    x   = Dense(64, activation="relu", name="dense1")(x)
    out = Dense(FORECAST_HORIZON * N_FEATURES, name="dense_out")(x)
    out = tf.keras.layers.Reshape(
            (FORECAST_HORIZON, N_FEATURES), name="output")(out)

    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="huber",       # robust to outliers
        metrics=["mae"]
    )
    return model

# ──────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN
# ──────────────────────────────────────────────────────────────────
def train_model(model, X_tr, y_tr, X_val, y_val, output_dir):
    print("\nSTEP 4 — Model Architecture")
    model.summary()

    print(f"\nSTEP 5 — Training  (epochs={EPOCHS}, batch={BATCH_SIZE})")
    callbacks = [
        # Stop when val_loss stops improving for 8 epochs
        EarlyStopping(
            monitor="val_loss", patience=8,
            restore_best_weights=True, verbose=1
        ),
        # Halve LR when val_loss plateaus for 4 epochs
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model_path = os.path.join(output_dir, "gnss_model.keras")
    model.save(model_path)
    print(f"  Model saved → {model_path}")
    return history

# ──────────────────────────────────────────────────────────────────
# STEP 6 — PREDICT DAY 8
# ──────────────────────────────────────────────────────────────────
def predict_day8(model, train_df, test_df, complete_sats, scaler):
    """
    For each satellite, feed the last 96 steps of Day 1–7
    into the model and get 96 predicted steps for Day 8.
    """
    print("\nSTEP 6 — Predicting Day 8 for all satellites")

    N_FEATURES  = len(TARGET_COLS)
    all_preds   = {}
    all_actuals = {}

    for sat_id in complete_sats:
        sat_train_sc = (train_df[train_df["Satellite_ID"] == sat_id]
                        [TARGET_COLS].values)
        sat_test_sc  = (test_df[test_df["Satellite_ID"]   == sat_id]
                        [TARGET_COLS].values)

        if len(sat_test_sc) != FORECAST_HORIZON:
            print(f"  [SKIP] {sat_id} — test has {len(sat_test_sc)} rows")
            continue

        # Last full day of training as input window
        last_window = sat_train_sc[-SEQ_LEN:].reshape(1, SEQ_LEN, N_FEATURES)
        pred_sc     = model.predict(last_window, verbose=0)[0]  # (96, 5)

        # Inverse transform back to original units
        all_preds[sat_id]   = scaler.inverse_transform(pred_sc)
        all_actuals[sat_id] = scaler.inverse_transform(sat_test_sc)

    print(f"  Predictions generated for {len(all_preds)} satellites")
    return all_preds, all_actuals

# ──────────────────────────────────────────────────────────────────
# STEP 7 — METRICS
# ──────────────────────────────────────────────────────────────────
def compute_metrics(all_preds, all_actuals):
    print("\nSTEP 7 — Computing Metrics")

    # ── Per-satellite, per-target metrics ──
    all_results = {}
    for sat_id in all_preds:
        metrics = {}
        for i, col in enumerate(TARGET_COLS):
            act  = all_actuals[sat_id][:, i]
            pred = all_preds[sat_id][:, i]
            metrics[col] = {
                "MAE":  float(mean_absolute_error(act, pred)),
                "RMSE": float(np.sqrt(mean_squared_error(act, pred)))
            }
        all_results[sat_id] = metrics

    # ── Aggregate across satellites ──
    agg = {col: {"MAE": [], "RMSE": []} for col in TARGET_COLS}
    for sat_id, mets in all_results.items():
        for col in TARGET_COLS:
            agg[col]["MAE"].append(mets[col]["MAE"])
            agg[col]["RMSE"].append(mets[col]["RMSE"])

    print(f"\n  {'Target':<22}  {'Mean MAE':>14}  {'Mean RMSE':>14}")
    print("  " + "─" * 54)
    for col in TARGET_COLS:
        print(f"  {col:<22}  "
              f"{np.mean(agg[col]['MAE']):>14.6f}  "
              f"{np.mean(agg[col]['RMSE']):>14.6f}")

    # ── Multi-horizon evaluation ──
    # Evaluates accuracy at each validity window specified by the problem
    HORIZONS = {
        "15 min":   1,
        "30 min":   2,
        "1 hour":   4,
        "2 hours":  8,
        "24 hours": 96
    }
    horizon_results = {}
    for label, steps in HORIZONS.items():
        h_mae = {col: [] for col in TARGET_COLS}
        for sat_id in all_preds:
            for i, col in enumerate(TARGET_COLS):
                h_mae[col].append(mean_absolute_error(
                    all_actuals[sat_id][:steps, i],
                    all_preds[sat_id][:steps, i]
                ))
        horizon_results[label] = {
            col: float(np.mean(h_mae[col])) for col in TARGET_COLS
        }

    print(f"\n  {'Horizon':<12}", end="")
    for col in TARGET_COLS:
        short = col.replace("Error_", "").replace("3D_Orbit_", "3D_")
        print(f"  {short:>12}", end="")
    print()
    print("  " + "─" * 80)
    for label, mets in horizon_results.items():
        print(f"  {label:<12}", end="")
        for col in TARGET_COLS:
            print(f"  {mets[col]:>12.6f}", end="")
        print()

    return all_results, agg, horizon_results

# ──────────────────────────────────────────────────────────────────
# STEP 8 — PLOTS
# ──────────────────────────────────────────────────────────────────
def generate_plots(history, all_preds, all_actuals,
                   all_results, horizon_results, output_dir):
    print("\nSTEP 8 — Generating Plots")

    HORIZONS   = {"15 min": 1, "30 min": 2, "1 hour": 4,
                  "2 hours": 8, "24 hours": 96}
    col_labels = [c.replace("Error_", "").replace("3D_Orbit_", "3D_")
                  for c in TARGET_COLS]
    sat_ids    = sorted(all_results.keys())

    # ── Plot 1: Training history ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")
    axes[0].plot(history.history["loss"],     label="Train", color="#1565C0")
    axes[0].plot(history.history["val_loss"], label="Val",   color="#E53935")
    axes[0].set_title("Huber Loss")
    axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["mae"],      label="Train", color="#1565C0")
    axes[1].plot(history.history["val_mae"],  label="Val",   color="#E53935")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, "01_training_history.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Plot 2: Prediction vs Actual — 3 GPS satellites ──────────
    sample_sats = [s for s in sorted(all_preds) if s.startswith("G")][:3]
    fig, axes   = plt.subplots(len(sample_sats), 2,
                               figsize=(16, 4 * len(sample_sats)))
    fig.suptitle("Day-8 Prediction vs Actual — 3 GPS Satellites",
                 fontsize=13, fontweight="bold")

    for row, sat in enumerate(sample_sats):
        for col_idx, (col_name, ylabel) in enumerate([
                ("3D_Orbit_Error", "3D Orbit Error (m)"),
                ("Error_Clock",    "Clock Error (s)")]):
            ax  = axes[row][col_idx]
            ci  = TARGET_COLS.index(col_name)
            act = all_actuals[sat][:, ci]
            prd = all_preds[sat][:, ci]
            ax.plot(act, color="#1565C0", lw=1.5, label="Actual")
            ax.plot(prd, color="#E53935", lw=1.5, ls="--", label="Predicted")
            mae = mean_absolute_error(act, prd)
            ax.set_title(f"{sat} — {ylabel}", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.annotate(f"MAE = {mae:.4f}",
                        xy=(0.02, 0.9), xycoords="axes fraction",
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    for col_idx in range(2):
        axes[-1][col_idx].set_xlabel("Time step (×15 min)")
    plt.tight_layout()
    p = os.path.join(output_dir, "02_prediction_vs_actual_GPS.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Plot 3: Prediction vs Actual — 3 GLONASS satellites ──────
    sample_sats_r = [s for s in sorted(all_preds) if s.startswith("R")][:3]
    if sample_sats_r:
        fig, axes = plt.subplots(len(sample_sats_r), 2,
                                 figsize=(16, 4 * len(sample_sats_r)))
        fig.suptitle("Day-8 Prediction vs Actual — 3 GLONASS Satellites",
                     fontsize=13, fontweight="bold")

        for row, sat in enumerate(sample_sats_r):
            for col_idx, (col_name, ylabel) in enumerate([
                    ("3D_Orbit_Error", "3D Orbit Error (m)"),
                    ("Error_Clock",    "Clock Error (s)")]):
                ax  = axes[row][col_idx]
                ci  = TARGET_COLS.index(col_name)
                act = all_actuals[sat][:, ci]
                prd = all_preds[sat][:, ci]
                ax.plot(act, color="#6A1B9A", lw=1.5, label="Actual")
                ax.plot(prd, color="#F57F17", lw=1.5, ls="--", label="Predicted")
                mae = mean_absolute_error(act, prd)
                ax.set_title(f"{sat} — {ylabel}", fontsize=10)
                ax.set_ylabel(ylabel, fontsize=8)
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
                ax.annotate(f"MAE = {mae:.4f}",
                            xy=(0.02, 0.9), xycoords="axes fraction",
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc="white", alpha=0.7))

        for col_idx in range(2):
            axes[-1][col_idx].set_xlabel("Time step (×15 min)")
        plt.tight_layout()
        p = os.path.join(output_dir, "03_prediction_vs_actual_GLONASS.png")
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")

    # ── Plot 4: Multi-horizon MAE heatmap ────────────────────────
    h_labels = list(HORIZONS.keys())
    mat      = np.array([[horizon_results[h][col]
                          for col in TARGET_COLS]
                         for h in h_labels])

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(TARGET_COLS)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(h_labels)))
    ax.set_yticklabels(h_labels, fontsize=11)
    ax.set_title("Multi-Horizon MAE Heatmap  (mean across all satellites)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="MAE")
    for r in range(len(h_labels)):
        for c in range(len(TARGET_COLS)):
            ax.text(c, r, f"{mat[r, c]:.4f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if mat[r, c] < mat.max() * 0.6 else "white")
    plt.tight_layout()
    p = os.path.join(output_dir, "04_multihorizon_mae_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Plot 5: Residual distributions (normality check) ─────────
    fig, axes = plt.subplots(1, len(TARGET_COLS), figsize=(22, 4))
    fig.suptitle(
        "Prediction Residuals — Normality Check  (pooled across all satellites)",
        fontsize=13, fontweight="bold")

    for i, (ax, col) in enumerate(zip(axes, TARGET_COLS)):
        resids = np.concatenate([
            all_actuals[s][:, i] - all_preds[s][:, i]
            for s in all_preds
        ])
        mu, sig = resids.mean(), resids.std()
        ax.hist(resids, bins=60, color="#5C6BC0",
                edgecolor="white", alpha=0.82, density=True)
        x = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
        ax.plot(x,
                np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi)),
                color="#E53935", lw=2,
                label=f"N(μ={mu:.2e}, σ={sig:.2e})")
        ax.set_title(col_labels[i], fontsize=10)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xlabel("Residual")

    plt.tight_layout()
    p = os.path.join(output_dir, "05_residual_distributions.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Plot 6: Per-satellite MAE bar chart ───────────────────────
    orbit_maes = [all_results[s]["3D_Orbit_Error"]["MAE"] for s in sat_ids]
    clock_maes = [all_results[s]["Error_Clock"]["MAE"]    for s in sat_ids]
    bar_colors = ["#1565C0" if s.startswith("G") else "#AD1457"
                  for s in sat_ids]

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    fig.suptitle("Per-Satellite MAE — Day-8 Prediction",
                 fontsize=13, fontweight="bold")

    for ax, vals, title, unit in [
            (axes[0], orbit_maes, "3D Orbit Error MAE (m)", "m"),
            (axes[1], clock_maes, "Clock Error MAE (s)",    "s")]:
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
            plt.Line2D([0], [0], color="black", ls="--",
                       label=f"Mean = {mean_val:.4f} {unit}")
        ], fontsize=8)

    plt.tight_layout()
    p = os.path.join(output_dir, "06_per_satellite_mae.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

# ──────────────────────────────────────────────────────────────────
# STEP 9 — SAVE METRICS JSON
# ──────────────────────────────────────────────────────────────────
def save_metrics(all_results, agg, horizon_results,
                 complete_sats, output_dir):
    summary = {
        "model": "Shared Bidirectional LSTM + GRU",
        "look_back_steps":    SEQ_LEN,
        "forecast_steps":     FORECAST_HORIZON,
        "satellites_trained": len(complete_sats),
        "satellites_evaluated": len(all_results),
        "aggregate_metrics": {
            col: {
                "Mean_MAE":  float(np.mean(agg[col]["MAE"])),
                "Mean_RMSE": float(np.mean(agg[col]["RMSE"]))
            }
            for col in TARGET_COLS
        },
        "multi_horizon_mae":    horizon_results,
        "per_satellite_metrics": all_results
    }
    p = os.path.join(output_dir, "metrics_summary.json")
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {p}")

# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Pipeline
    train_df, test_df, complete_sats = load_and_clean(args.data)
    train_df, test_df, scaler        = scale_data(train_df, test_df)
    X_tr, y_tr, X_val, y_val        = build_all_sequences(train_df, complete_sats)

    model   = build_model()
    history = train_model(model, X_tr, y_tr, X_val, y_val, args.output)

    all_preds, all_actuals           = predict_day8(
                                           model, train_df, test_df,
                                           complete_sats, scaler)
    all_results, agg, horizon_results = compute_metrics(all_preds, all_actuals)
    generate_plots(history, all_preds, all_actuals,
                   all_results, horizon_results, args.output)
    save_metrics(all_results, agg, horizon_results, complete_sats, args.output)

    print("\n" + "=" * 65)
    print("COMPLETE ✓  All outputs saved to:", args.output)
    print("=" * 65)


if __name__ == "__main__":
    main()

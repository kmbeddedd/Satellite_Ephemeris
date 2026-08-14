"""
Enhanced CLI Entrypoint for Training & Evaluating the Multi-Horizon BiLSTM + GRU Model
Optimized for GPU acceleration (CUDA), Cosine Learning Rate Scheduling, and Residual Anchoring.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    SEQ_LEN,
    FORECAST_HORIZON,
    TARGET_COLS_5,
    KERAS_DEFAULTS,
    DEFAULT_SEED
)
from src.data import load_and_clean_data, scale_datasets_keras, build_keras_sequences
from src.models.pytorch_bilstm import BiLSTMGRUPyTorchModel
from src.evaluate import (
    compute_aggregate_metrics,
    compute_multi_horizon_metrics,
    print_metrics_summary,
    save_metrics_summary
)
from src.visualize import (
    plot_training_history,
    plot_prediction_vs_actual,
    plot_multihorizon_heatmap,
    plot_residual_distributions,
    plot_per_satellite_mae
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Enhanced BiLSTM + GRU GNSS Forecaster (GPU & CPU)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.8e-3, help="Learning rate")
    parser.add_argument("--bilstm-units", type=int, default=64, help="BiLSTM hidden units")
    parser.add_argument("--gru-units", type=int, default=64, help="GRU hidden units")
    parser.add_argument("--dropout-1", type=float, default=0.2, help="Dropout 1 rate")
    parser.add_argument("--dropout-2", type=float, default=0.1, help="Dropout 2 rate")
    parser.add_argument("--backend", choices=["auto", "keras", "torch"], default="auto", help="Deep learning backend")
    parser.add_argument("--device", default="auto", help="Target device: 'cuda', 'cuda:0', 'cpu', or 'auto'")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def train_pytorch(args, train_df_scaled, test_df_scaled, complete_sats, scaler):
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    device = torch.device(device_str)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"  Target Compute Device: GPU -> {gpu_name} ({total_vram:.2f} GB VRAM)")
    else:
        print("  Target Compute Device: CPU")

    # 1. Build sequence datasets
    X_train, y_train, X_val, y_val = build_keras_sequences(
        train_df_scaled, complete_sats, seq_len=SEQ_LEN, horizon=FORECAST_HORIZON, target_cols=TARGET_COLS_5, seed=args.seed
    )

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    # 2. Instantiate Enhanced Model
    model = BiLSTMGRUPyTorchModel(
        seq_len=SEQ_LEN,
        n_features=len(TARGET_COLS_5),
        forecast_horizon=FORECAST_HORIZON,
        bilstm_units=args.bilstm_units,
        gru_units=args.gru_units,
        dropout_1=args.dropout_1,
        dropout_2=args.dropout_2
    ).to(device)

    # Physics-informed composite loss
    huber_criterion = nn.SmoothL1Loss(beta=0.5)

    def composite_loss(preds, targets):
        base_huber = huber_criterion(preds, targets)
        # Smoothness penalty on predicted trajectory acceleration
        diffs = preds[:, 1:, :] - preds[:, :-1, :]
        smooth_loss = torch.mean(diffs ** 2)
        return base_huber + 0.02 * smooth_loss

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    print(f"\nStarting Enhanced BiLSTM + GRU Training on {device} (epochs={args.epochs}, batch={args.batch_size})...")

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_mae_sum = 0.0, 0.0
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(bx)
            loss = composite_loss(out, by)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(bx)
            train_mae_sum += torch.mean(torch.abs(out - by)).item() * len(bx)

        epoch_loss = train_loss / len(train_ds)
        epoch_mae = train_mae_sum / len(train_ds)

        model.eval()
        val_loss, val_mae_sum = 0.0, 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                out = model(bx)
                loss = composite_loss(out, by)
                val_loss += loss.item() * len(bx)
                val_mae_sum += torch.mean(torch.abs(out - by)).item() * len(bx)

        epoch_val_loss = val_loss / len(val_ds)
        epoch_val_mae = val_mae_sum / len(val_ds)

        scheduler.step(epoch_val_loss)

        history["loss"].append(epoch_loss)
        history["val_loss"].append(epoch_val_loss)
        history["mae"].append(epoch_mae)
        history["val_mae"].append(epoch_val_mae)

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {epoch_loss:.5f} | Val Loss: {epoch_val_loss:.5f} | Val MAE: {epoch_val_mae:.5f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 8:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model_path = os.path.join(args.output, "gnss_bilstm_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved successfully -> {model_path}")

    # Inference on Out-of-Sample Day 8
    n_features = len(TARGET_COLS_5)
    all_preds, all_actuals = {}, {}
    model.eval()

    with torch.no_grad():
        for sat_id in complete_sats:
            sat_train_sc = train_df_scaled[train_df_scaled["Satellite_ID"] == sat_id][TARGET_COLS_5].values
            sat_test_sc = test_df_scaled[test_df_scaled["Satellite_ID"] == sat_id][TARGET_COLS_5].values
            if len(sat_test_sc) != FORECAST_HORIZON:
                continue
            last_window = sat_train_sc[-SEQ_LEN:].reshape(1, SEQ_LEN, n_features)
            t_inp = torch.tensor(last_window, dtype=torch.float32, device=device)
            pred_sc = model(t_inp).cpu().numpy()[0]
            all_preds[sat_id] = scaler.inverse_transform(pred_sc)
            all_actuals[sat_id] = scaler.inverse_transform(sat_test_sc)

    return history, all_preds, all_actuals


def run_training():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)

    # 1. Load and clean data
    train_df, test_df, complete_sats = load_and_clean_data(args.data)

    # 2. Scale datasets
    train_df_scaled, test_df_scaled, scaler = scale_datasets_keras(train_df, test_df, TARGET_COLS_5)

    # Train on GPU with PyTorch
    history, all_preds, all_actuals = train_pytorch(args, train_df_scaled, test_df_scaled, complete_sats, scaler)

    # 3. Evaluate Performance
    per_sat_results, aggregate = compute_aggregate_metrics(all_actuals, all_preds, TARGET_COLS_5)
    horizon_results = compute_multi_horizon_metrics(all_actuals, all_preds, TARGET_COLS_5)
    print_metrics_summary(aggregate, horizon_results, TARGET_COLS_5)

    # 4. Save Metrics
    save_metrics_summary(
        filepath=os.path.join(args.output, "metrics_summary.json"),
        model_name="Enhanced BiLSTM + GRU Forecaster (Residual Anchor)",
        aggregate=aggregate,
        horizon_results=horizon_results,
        per_sat_results=per_sat_results,
        extra_meta={
            "satellites_trained": len(complete_sats),
            "satellites_evaluated": len(all_preds),
            "lookback_steps": SEQ_LEN,
            "forecast_steps": FORECAST_HORIZON,
            "improvements": "Residual Anchor Skip-Connection + Attention Context Pooling + Huber-Smoothness Loss"
        }
    )

    # 5. Generate Diagnostic Visualizations
    print("\nGenerating Diagnostic Figures...")
    plot_training_history(history, os.path.join(args.output, "01_training_history.png"))

    gps_sats = [s for s in sorted(all_preds) if s.startswith("G")][:3]
    glo_sats = [s for s in sorted(all_preds) if s.startswith("R")][:3]

    if gps_sats:
        plot_prediction_vs_actual(all_actuals, all_preds, gps_sats, TARGET_COLS_5, os.path.join(args.output, "02_prediction_vs_actual_GPS.png"), "GPS (Day 8)")
    if glo_sats:
        plot_prediction_vs_actual(all_actuals, all_preds, glo_sats, TARGET_COLS_5, os.path.join(args.output, "03_prediction_vs_actual_GLONASS.png"), "GLONASS (Day 8)")

    plot_multihorizon_heatmap(horizon_results, TARGET_COLS_5, os.path.join(args.output, "04_multihorizon_mae_heatmap.png"))
    plot_residual_distributions(all_actuals, all_preds, TARGET_COLS_5, os.path.join(args.output, "05_residual_distributions.png"))
    plot_per_satellite_mae(per_sat_results, TARGET_COLS_5, os.path.join(args.output, "06_per_satellite_mae.png"))

    print(f"\nTraining & Evaluation Pipeline Complete! All artifacts saved to: {args.output}")


if __name__ == "__main__":
    run_training()

"""
CLI Entrypoint for Optuna Bayesian Hyperparameter Optimization
Supports GPU (CUDA) and CPU execution environments.
"""

import argparse
import os
import optuna
import numpy as np

from src.config import (
    DEFAULT_DATA_PATH,
    SEQ_LEN,
    FORECAST_HORIZON,
    TARGET_COLS_5,
    DEFAULT_SEED
)
from src.data import load_and_clean_data, scale_datasets_keras, build_keras_sequences

# Global data placeholders for objective function
_X_tr, _y_tr, _X_val, _y_val = None, None, None, None
_backend = "torch"
_device_str = "auto"


def objective_keras(trial: optuna.Trial) -> float:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from src.models.keras_bilstm import build_bilstm_gru_model

    bilstm_units = trial.suggest_categorical("bilstm_units", [32, 64, 128])
    gru_units = trial.suggest_categorical("gru_units", [16, 32, 64])
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.4)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.4)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    model = build_bilstm_gru_model(
        seq_len=SEQ_LEN,
        n_features=len(TARGET_COLS_5),
        forecast_horizon=FORECAST_HORIZON,
        bilstm_units=bilstm_units,
        gru_units=gru_units,
        dropout_1=dropout_1,
        dropout_2=dropout_2,
        learning_rate=lr
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=0)
    ]

    history = model.fit(
        _X_tr, _y_tr,
        validation_data=(_X_val, _y_val),
        epochs=15,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    val_mae = min(history.history["val_mae"])
    return float(val_mae)


def objective_pytorch(trial: optuna.Trial) -> float:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from src.models.pytorch_bilstm import BiLSTMGRUPyTorchModel

    if _device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(_device_str)

    bilstm_units = trial.suggest_categorical("bilstm_units", [32, 64, 128])
    gru_units = trial.suggest_categorical("gru_units", [16, 32, 64])
    dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.4)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.4)
    lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])

    train_ds = TensorDataset(torch.tensor(_X_tr, dtype=torch.float32), torch.tensor(_y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(_X_val, dtype=torch.float32), torch.tensor(_y_val, dtype=torch.float32))

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    model = BiLSTMGRUPyTorchModel(
        seq_len=SEQ_LEN,
        n_features=len(TARGET_COLS_5),
        forecast_horizon=FORECAST_HORIZON,
        bilstm_units=bilstm_units,
        gru_units=gru_units,
        dropout_1=dropout_1,
        dropout_2=dropout_2
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float("inf")

    for epoch in range(12):  # Fast tuning loop
        model.train()
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        val_mae_sum = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                out = model(bx)
                val_mae_sum += torch.mean(torch.abs(out - by)).item() * len(bx)

        epoch_val_mae = val_mae_sum / len(val_ds)
        if epoch_val_mae < best_val_mae:
            best_val_mae = epoch_val_mae

    return float(best_val_mae)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for GNSS Forecaster (GPU & CPU)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--n-trials", type=int, default=8, help="Number of Optuna trials")
    parser.add_argument("--backend", choices=["auto", "keras", "torch"], default="auto", help="Execution backend")
    parser.add_argument("--device", default="auto", help="Compute device ('cuda', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def run_tuning():
    global _X_tr, _y_tr, _X_val, _y_val, _backend, _device_str
    args = parse_args()
    np.random.seed(args.seed)

    _backend = args.backend
    _device_str = args.device

    if _backend == "auto":
        try:
            import tensorflow
            _backend = "keras"
        except ImportError:
            _backend = "torch"

    print(f"Hyperparameter Tuning Backend: {_backend.upper()} (device={_device_str})")
    print("Loading and preparing sequences for hyperparameter optimization...")
    train_df, test_df, complete_sats = load_and_clean_data(args.data)
    train_df_sc, test_df_sc, _ = scale_datasets_keras(train_df, test_df, TARGET_COLS_5)

    _X_tr, _y_tr, _X_val, _y_val = build_keras_sequences(
        train_df_sc,
        complete_sats,
        seq_len=SEQ_LEN,
        horizon=FORECAST_HORIZON,
        target_cols=TARGET_COLS_5,
        seed=args.seed
    )

    print(f"\nStarting Optuna Study ({args.n_trials} trials)...")
    study = optuna.create_study(direction="minimize")

    target_objective = objective_keras if _backend == "keras" else objective_pytorch
    study.optimize(target_objective, n_trials=args.n_trials)

    print("\n" + "=" * 50)
    print("OPTUNA HYPERPARAMETER TUNING COMPLETE")
    print("=" * 50)
    trial = study.best_trial
    print(f"  Best Validation MAE: {trial.value:.5f}")
    print("  Optimal Hyperparameters:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    run_tuning()

"""Seeded Optuna tuning on the same leakage-safe folds used for training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.artifacts import set_reproducible_seed
from src.config import DEFAULT_DATA_PATH, DEFAULT_SEED, FORECAST_HORIZON, SEQ_LEN
from src.data import prepare_pytorch_datasets
from src.models.pytorch_bilstm import BiLSTMGRUPyTorchModel


def _masked_huber(prediction: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, actual, beta=0.5, reduction="none")
    weights = mask.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_mae(prediction: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(prediction.dtype)
    return (torch.abs(prediction - actual) * weights).sum() / weights.sum().clamp_min(1.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune the masked GNSS BiLSTM benchmark")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--n-trials", "--trials", dest="n_trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--backend", choices=("auto", "torch", "keras"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default="tuning_results.json")
    args = parser.parse_args(argv)
    if args.backend == "keras":
        parser.error("Keras tuning is disabled because the compatibility model cannot apply target masks")
    if args.n_trials < 1 or args.epochs < 1:
        parser.error("n-trials and epochs must be positive")
    return args


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def run_tuning(argv: list[str] | None = None) -> optuna.Study:
    args = parse_args(argv)
    device = _device(args.device)
    set_reproducible_seed(args.seed, deterministic=True)
    bundle = prepare_pytorch_datasets(
        args.data,
        input_window=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        batch_size=64,
        seed=args.seed,
    )
    train_tensors = TensorDataset(
        torch.as_tensor(bundle["X_train"], dtype=torch.float32),
        torch.as_tensor(bundle["Y_train"], dtype=torch.float32),
        torch.as_tensor(bundle["TARGET_MASK_train"], dtype=torch.float32),
    )
    val_tensors = TensorDataset(
        torch.as_tensor(bundle["X_val"], dtype=torch.float32),
        torch.as_tensor(bundle["Y_val"], dtype=torch.float32),
        torch.as_tensor(bundle["TARGET_MASK_val"], dtype=torch.float32),
    )

    def objective(trial: optuna.Trial) -> float:
        trial_seed = args.seed + trial.number
        set_reproducible_seed(trial_seed, deterministic=True)
        bilstm_units = trial.suggest_categorical("bilstm_units", [32, 64, 96])
        gru_units = trial.suggest_categorical("gru_units", [32, 64, 96])
        dropout_1 = trial.suggest_float("dropout_1", 0.1, 0.4)
        dropout_2 = trial.suggest_float("dropout_2", 0.05, 0.3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        generator = torch.Generator().manual_seed(trial_seed)
        train_loader = DataLoader(
            train_tensors,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_tensors,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=device.type == "cuda",
        )
        model = BiLSTMGRUPyTorchModel(
            seq_len=SEQ_LEN,
            n_features=bundle["num_features"],
            output_dim=bundle["output_dim"],
            target_feature_indices=tuple(bundle["target_feature_indices"]),
            forecast_horizon=FORECAST_HORIZON,
            bilstm_units=bilstm_units,
            gru_units=gru_units,
            dropout_1=dropout_1,
            dropout_2=dropout_2,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        best_validation = float("inf")
        for epoch in range(args.epochs):
            model.train()
            for x, actual, mask in train_loader:
                x = x.to(device, non_blocking=True)
                actual = actual.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = _masked_huber(model(x), actual, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            numerator = denominator = 0.0
            with torch.no_grad():
                for x, actual, mask in val_loader:
                    x = x.to(device, non_blocking=True)
                    actual = actual.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    prediction = model(x)
                    numerator += float((torch.abs(prediction - actual) * mask).sum())
                    denominator += float(mask.sum())
            validation_mae = numerator / max(denominator, 1.0)
            best_validation = min(best_validation, validation_mae)
            trial.report(validation_mae, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(best_validation)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(3, args.n_trials))
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=args.n_trials)
    payload = {
        "objective": "masked validation MAE in standardized target space",
        "seed": args.seed,
        "split_metadata": bundle["split_metadata"],
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
            }
            for trial in study.trials
        ],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"Best validation MAE: {study.best_value:.6f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Report written to {destination.resolve()}")
    return study


if __name__ == "__main__":
    run_tuning()

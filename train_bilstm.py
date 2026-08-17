"""Train the deterministic BiLSTM-GRU benchmark on leakage-safe GNSS folds."""

from __future__ import annotations

import argparse
import copy
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from audit_data import audit_csv
from src.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    current_git_sha,
    scaler_to_state,
    set_reproducible_seed,
    sha256_file,
    write_json,
)
from src.baselines import generate_baseline_forecasts
from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    FORECAST_HORIZON,
    HORIZON_MAP,
    SEQ_LEN,
)
from src.data import FastGPUTensorLoader, prepare_pytorch_datasets
from src.evaluate import (
    compare_candidate_to_baseline,
    compute_tensor_horizon_metrics,
    evaluate_forecasts,
)
from src.models.pytorch_bilstm import BiLSTMGRUPyTorchModel
from src.visualize import (
    plot_multihorizon_heatmap,
    plot_per_satellite_mae,
    plot_prediction_vs_actual,
    plot_residual_distributions,
    plot_training_history,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the masked BiLSTM-GRU GNSS benchmark")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--bilstm-units", type=int, default=64)
    parser.add_argument("--gru-units", type=int, default=64)
    parser.add_argument("--dropout-1", type=float, default=0.2)
    parser.add_argument("--dropout-2", type=float, default=0.1)
    parser.add_argument("--backend", choices=("auto", "torch", "keras"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.backend == "keras":
        parser.error(
            "The Keras compatibility model does not consume target-availability masks. "
            "Use --backend torch for scientifically valid training."
        )
    if min(args.epochs, args.batch_size, args.early_stopping_patience) < 1:
        parser.error("epochs, batch-size, and early-stopping-patience must be positive")
    return args


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _loader(
    bundle: dict[str, Any], split: str, device: torch.device, batch_size: int, shuffle: bool
) -> FastGPUTensorLoader:
    tensors = (
        torch.as_tensor(bundle[f"X_{split}"], dtype=torch.float32, device=device),
        torch.as_tensor(bundle[f"Y_{split}"], dtype=torch.float32, device=device),
        torch.as_tensor(bundle[f"TARGET_MASK_{split}"], dtype=torch.float32, device=device),
    )
    return FastGPUTensorLoader(tensors, batch_size=batch_size, shuffle=shuffle, device=device)


def _masked_huber(predicted: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    element_loss = F.smooth_l1_loss(predicted, actual, beta=0.5, reduction="none")
    weights = mask.to(dtype=element_loss.dtype)
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


def _masked_mae(predicted: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=predicted.dtype)
    return (torch.abs(predicted - actual) * weights).sum() / weights.sum().clamp_min(1.0)


def _run_epoch(
    model: BiLSTMGRUPyTorchModel,
    loader: FastGPUTensorLoader,
    optimizer: torch.optim.Optimizer | None,
    amp_scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_mae = 0.0
    observations = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for x, actual, mask in loader:
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted = model(x)
                loss = _masked_huber(predicted, actual, mask)
                mae = _masked_mae(predicted, actual, mask)
            if training:
                if use_amp:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            total_mae += float(mae.detach()) * len(x)
            observations += len(x)
    return total_loss / max(observations, 1), total_mae / max(observations, 1)


@torch.no_grad()
def _predict(
    model: BiLSTMGRUPyTorchModel, loader: FastGPUTensorLoader, use_amp: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predicted, actual, masks = [], [], []
    for x, target, mask in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(x)
        predicted.append(output.float().cpu().numpy())
        actual.append(target.float().cpu().numpy())
        masks.append(mask.bool().cpu().numpy())
    return np.concatenate(predicted), np.concatenate(actual), np.concatenate(masks)


def _promotion_rules(target_cols: list[str]) -> list[dict[str, Any]]:
    rules = []
    for label in HORIZON_MAP:
        for target in target_cols:
            rules.append({
                "metric": f"horizons.{label}.exact_lead.per_target.{target}.mae",
                "direction": "lower",
            })
        rules.append({
            "metric": f"horizons.{label}.exact_lead.orbit_3d_vector_error.mae",
            "direction": "lower",
        })
    return rules


def _real_history(bundle: dict[str, Any], split: str) -> np.ndarray:
    indices = np.asarray(bundle["target_feature_indices"], dtype=np.int64)
    scaled = bundle[f"X_{split}"][:, :, indices]
    scaler = bundle["feature_scaler"]
    return (
        scaled * np.asarray(scaler.scale_)[indices][None, None, :]
        + np.asarray(scaler.mean_)[indices][None, None, :]
    )


def _plot_views(
    output: Path,
    history: dict[str, list[float]],
    actual: np.ndarray,
    predicted: np.ndarray,
    satellite_ids: np.ndarray,
    target_cols: list[str],
    point_report: dict[str, Any],
) -> None:
    plot_training_history(history, str(output / "01_training_history.png"))
    actual_by_sat, predicted_by_sat = {}, {}
    for index, satellite in enumerate(satellite_ids):
        actual_by_sat.setdefault(str(satellite), actual[index])
        predicted_by_sat.setdefault(str(satellite), predicted[index])
    gps = [sat for sat in sorted(actual_by_sat) if sat.startswith("G")][:3]
    glo = [sat for sat in sorted(actual_by_sat) if sat.startswith("R")][:3]
    if gps:
        plot_prediction_vs_actual(
            actual_by_sat, predicted_by_sat, gps, target_cols,
            str(output / "02_prediction_vs_actual_GPS.png"), "GPS test forecasts",
        )
    if glo:
        plot_prediction_vs_actual(
            actual_by_sat, predicted_by_sat, glo, target_cols,
            str(output / "03_prediction_vs_actual_GLONASS.png"), "GLONASS test forecasts",
        )
    horizons = compute_tensor_horizon_metrics(actual, predicted, target_cols, HORIZON_MAP)
    plot_multihorizon_heatmap(horizons, target_cols, str(output / "04_multihorizon_mae_heatmap.png"))
    plot_residual_distributions(
        actual_by_sat, predicted_by_sat, target_cols, str(output / "05_residual_distributions.png")
    )
    per_sat = {
        sat: {
            target: {
                "MAE": point_report["slices"]["per_satellite"][sat]
                ["all_forecast_points"]["per_target"][target]["mae"]
            }
            for target in target_cols
        }
        for sat in actual_by_sat
    }
    plot_per_satellite_mae(per_sat, target_cols, str(output / "06_per_satellite_mae.png"))


def run_training(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(args.seed, deterministic=not args.nondeterministic)
    device = _device(args.device)
    use_amp = device.type == "cuda"
    print(f"Compute device: {device}; deterministic={not args.nondeterministic}")

    bundle = prepare_pytorch_datasets(
        args.data,
        input_window=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader = _loader(bundle, "train", device, args.batch_size, True)
    val_loader = _loader(bundle, "val", device, args.batch_size, False)
    test_loader = _loader(bundle, "test", device, args.batch_size, False)
    model_config = {
        "seq_len": SEQ_LEN,
        "n_features": bundle["num_features"],
        "output_dim": bundle["output_dim"],
        "target_feature_indices": tuple(bundle["target_feature_indices"]),
        "forecast_horizon": FORECAST_HORIZON,
        "bilstm_units": args.bilstm_units,
        "gru_units": args.gru_units,
        "dropout_1": args.dropout_1,
        "dropout_2": args.dropout_2,
    }
    model = BiLSTMGRUPyTorchModel(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"loss": [], "val_loss": [], "mae": [], "val_mae": []}
    best_loss, best_epoch, best_state, best_optimizer_state, patience = (
        float("inf"), -1, None, None, 0
    )
    for epoch in range(args.epochs):
        train_loss, train_mae = _run_epoch(model, train_loader, optimizer, amp_scaler, use_amp)
        val_loss, val_mae = _run_epoch(model, val_loader, None, amp_scaler, use_amp)
        scheduler.step(val_loss)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} train={train_loss:.6f} "
            f"val={val_loss:.6f} val_mae={val_mae:.6f}"
        )
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}.")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)

    predicted_scaled, actual_scaled, mask = _predict(model, test_loader, use_amp)
    scaler = bundle["target_scaler"]
    output_dim = bundle["output_dim"]
    predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, output_dim)).reshape(predicted_scaled.shape)
    actual = scaler.inverse_transform(actual_scaled.reshape(-1, output_dim)).reshape(actual_scaled.shape)
    predicted = np.where(mask, predicted, np.nan)
    actual = np.where(mask, actual, np.nan)
    satellite_ids = np.asarray(bundle["SATELLITE_IDS_test"], dtype=str)
    point_report = evaluate_forecasts(
        actual,
        predicted,
        bundle["target_cols"],
        satellite_ids=satellite_ids,
        valid_mask=mask,
    )

    baseline_reports, promotion = {}, {}
    baseline_predictions = generate_baseline_forecasts(
        _real_history(bundle, "test"), horizon=FORECAST_HORIZON, season_length=SEQ_LEN
    )
    for name, baseline_prediction in baseline_predictions.items():
        baseline_prediction = np.where(mask, baseline_prediction, np.nan)
        report = evaluate_forecasts(
            actual,
            baseline_prediction,
            bundle["target_cols"],
            satellite_ids=satellite_ids,
            valid_mask=mask,
        )
        baseline_reports[name] = report
        promotion[name] = compare_candidate_to_baseline(
            point_report, report, _promotion_rules(bundle["target_cols"]), require_all=True
        )
    quality = audit_csv(args.data)
    report = {
        "model": point_report,
        "baselines": baseline_reports,
        "promotion_against_each_baseline": promotion,
        "promotion_eligible": bool(
            quality["passed"] and promotion and all(item["passed"] for item in promotion.values())
        ),
        "data_quality": quality,
        "coverage": {
            "training_vocabulary_satellites": len(bundle["satellite_classes"]),
            "test_satellites_evaluated": len(set(satellite_ids.tolist())),
            "test_samples": int(len(satellite_ids)),
            "valid_target_fraction": float(mask.mean()),
        },
    }
    write_json(output / "evaluation_report.json", report)
    np.savez_compressed(
        output / "test_predictions.npz",
        actual=actual,
        prediction=predicted,
        valid_mask=mask,
        satellite_ids=satellite_ids,
        label_timestamps=bundle["LABEL_TIMESTAMPS_test"],
    )
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_class": "BiLSTMGRUPyTorchModel",
        "model_config": model_config,
        "training_config": vars(args),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": best_optimizer_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "target_cols": bundle["target_cols"],
        "feature_cols": bundle["feature_cols"],
        "target_feature_indices": bundle["target_feature_indices"],
        "satellite_classes": bundle["satellite_classes"],
        "feature_scaler": scaler_to_state(bundle["feature_scaler"]),
        "target_scaler": scaler_to_state(bundle["target_scaler"]),
        "split_metadata": bundle["split_metadata"],
        "data_quality_report": bundle["data_quality_report"],
        "data_sha256": sha256_file(args.data),
        "git_sha": current_git_sha(Path(__file__).resolve().parent),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    torch.save(artifact, output / "gnss_bilstm_bundle.pt")
    manifest = {key: value for key, value in artifact.items() if not key.endswith("state_dict")}
    write_json(output / "artifact_manifest.json", manifest)
    if not args.skip_plots:
        _plot_views(output, history, actual, predicted, satellite_ids, bundle["target_cols"], point_report)
    print(f"Artifacts written to {output.resolve()}")
    print(f"Promotion eligible: {report['promotion_eligible']}")
    return report


if __name__ == "__main__":
    run_training()

"""Train and evaluate the probabilistic GNSS sequence forecaster.

The pipeline consumes leakage-safe, masked temporal folds from ``src.data``;
selects the best validation checkpoint; records preprocessing/provenance; scores
simple baselines; and calibrates uncertainty on the validation fold. Diffusion
is an optional residual-model ablation and is evaluated over every test series.
"""

from __future__ import annotations

import argparse
import copy
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from src.calibration import evaluate_conformal_intervals, fit_scaled_conformal
from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_SEED,
    DIFFUSION_DEFAULTS,
    FORECAST_HORIZON,
    HORIZON_MAP,
    SEQ_LEN,
    TRANSFORMER_DEFAULTS,
)
from src.data import FastGPUTensorLoader, prepare_pytorch_datasets
from src.evaluate import (
    compare_candidate_to_baseline,
    compute_probabilistic_metrics,
    compute_sample_metrics,
    compute_tensor_horizon_metrics,
    evaluate_forecasts,
)
from src.models.losses import composite_transformer_loss, diffusion_mse_loss
from src.models.pytorch_diffusion import (
    ConditionalDiffusionDenoiser,
    DiffusionSchedule,
    sample_ddim_forecast,
)
from src.models.pytorch_transformer import GNSSForecaster
from src.visualize import (
    plot_diffusion_samples,
    plot_frequency_spectrum,
    plot_multihorizon_heatmap,
    plot_probabilistic_uncertainty,
    plot_training_history,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a leakage-safe probabilistic GNSS forecaster")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output", default="./transformer_results")
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_DEFAULTS["epochs"])
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--diffusion-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--bilstm-units", type=int, default=48)
    parser.add_argument("--gru-units", type=int, default=48)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--distribution", choices=("student_t", "gaussian"), default="student_t")
    parser.add_argument("--student-t-df", type=float, default=3.0)
    parser.add_argument("--use-revin", action="store_true", help="Enable RevIN as an explicit ablation")
    parser.add_argument("--no-revin", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable-event-head", action="store_true", help="Requires externally sourced event labels")
    parser.add_argument("--lambda-event", type=float, default=0.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.0)
    parser.add_argument("--lambda-dilate", type=float, default=0.0)
    parser.add_argument("--enable-diffusion", action="store_true")
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--diffusion-eval-samples", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.early_stopping_patience < 1:
        parser.error("epochs, batch-size, and early-stopping-patience must be positive")
    if args.student_t_df <= 2:
        parser.error("--student-t-df must be greater than 2")
    if args.no_revin:
        args.use_revin = False
    if args.enable_event_head:
        parser.error(
            "No externally sourced maneuver/clock-event labels are present; "
            "the old target-threshold pseudo-event task is intentionally disabled."
        )
    return args


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _gpu_loader(
    bundle: dict[str, Any], split: str, device: torch.device, batch_size: int, shuffle: bool
) -> FastGPUTensorLoader:
    tensors = (
        torch.as_tensor(bundle[f"X_{split}"], dtype=torch.float32, device=device),
        torch.as_tensor(bundle[f"Y_{split}"], dtype=torch.float32, device=device),
        torch.as_tensor(bundle[f"SAT_{split}"], dtype=torch.long, device=device),
        torch.as_tensor(bundle[f"SPIKE_{split}"], dtype=torch.float32, device=device),
        torch.as_tensor(bundle[f"TARGET_MASK_{split}"], dtype=torch.float32, device=device),
    )
    return FastGPUTensorLoader(tensors, batch_size=batch_size, shuffle=shuffle, device=device)


def _loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    spikes: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    mu, scale, event_logits, _ = outputs
    return composite_transformer_loss(
        mu=mu,
        sigma=scale,
        spike_probs=event_logits,
        targets=targets,
        spike_targets=spikes,
        target_mask=mask,
        distribution=args.distribution,
        degrees_of_freedom=args.student_t_df,
        lambda_spike=args.lambda_event,
        lambda_smooth=args.lambda_smooth,
        lambda_dilate=args.lambda_dilate,
    )


def train_one_epoch(
    model: GNSSForecaster,
    loader: FastGPUTensorLoader,
    optimizer: torch.optim.Optimizer,
    amp_scaler: torch.amp.GradScaler,
    use_amp: bool,
    args: argparse.Namespace,
) -> float:
    model.train()
    total, observations = 0.0, 0
    for x, y, sat, spikes, mask in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = _loss(model(x, sat), y, spikes, mask, args)
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
        total += float(loss.detach()) * len(x)
        observations += len(x)
    return total / max(observations, 1)


@torch.no_grad()
def validate_epoch(
    model: GNSSForecaster,
    loader: FastGPUTensorLoader,
    use_amp: bool,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.eval()
    total, observations, mean_scales = 0.0, 0, []
    for x, y, sat, spikes, mask in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(x, sat)
            loss = _loss(outputs, y, spikes, mask, args)
        total += float(loss) * len(x)
        observations += len(x)
        valid_scale = outputs[1][mask.bool()]
        if valid_scale.numel():
            mean_scales.append(float(valid_scale.mean()))
    return total / max(observations, 1), float(np.mean(mean_scales))


@torch.no_grad()
def _collect_predictions(
    model: GNSSForecaster, loader: FastGPUTensorLoader, use_amp: bool
) -> dict[str, np.ndarray]:
    model.eval()
    values = {key: [] for key in ("mean", "scale", "actual", "mask", "sat")}
    for x, y, sat, _, mask in loader:
        with torch.amp.autocast("cuda", enabled=use_amp):
            mean, scale, _, _ = model(x, sat)
        values["mean"].append(mean.float().cpu().numpy())
        values["scale"].append(scale.float().cpu().numpy())
        values["actual"].append(y.float().cpu().numpy())
        values["mask"].append(mask.bool().cpu().numpy())
        values["sat"].append(sat.cpu().numpy())
    return {key: np.concatenate(items, axis=0) for key, items in values.items()}


def _inverse_predictions(collected: dict[str, np.ndarray], target_scaler) -> dict[str, np.ndarray]:
    target_dim = collected["actual"].shape[-1]
    mean = target_scaler.inverse_transform(collected["mean"].reshape(-1, target_dim)).reshape(collected["mean"].shape)
    actual = target_scaler.inverse_transform(collected["actual"].reshape(-1, target_dim)).reshape(collected["actual"].shape)
    scale = collected["scale"] * np.asarray(target_scaler.scale_)[None, None, :]
    mask = collected["mask"].astype(bool)
    return {
        **collected,
        "actual_real": np.where(mask, actual, np.nan),
        "mean_real": np.where(mask, mean, np.nan),
        "scale_real": np.where(mask, scale, np.nan),
    }


def _model_config(args: argparse.Namespace, bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_features": bundle["num_features"],
        "num_satellites": bundle["num_satellites"],
        "d_model": args.d_model,
        "bilstm_units": args.bilstm_units,
        "gru_units": args.gru_units,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "seq_len": SEQ_LEN,
        "forecast_horizon": FORECAST_HORIZON,
        "output_dim": bundle["output_dim"],
        "target_feature_indices": tuple(bundle["target_feature_indices"]),
        "use_revin": args.use_revin,
        "enable_event_head": False,
        "dropout": args.dropout,
    }


def _artifact_metadata(args: argparse.Namespace, bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_class": "GNSSForecaster",
        "model_config": _model_config(args, bundle),
        "training_config": vars(args),
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


def _calibration_gate(
    conformal_report: dict[str, Any],
    required_coverages: tuple[float, ...] = (0.8, 0.9, 0.95),
    tolerance: float = 0.05,
) -> dict[str, Any]:
    checks = []
    for target, target_report in conformal_report["per_target"].items():
        for coverage in required_coverages:
            key = f"{coverage:.6g}"
            metrics = target_report.get(key, {})
            empirical = metrics.get("empirical_coverage")
            error = None if empirical is None else abs(float(empirical) - coverage)
            checks.append({
                "target": target,
                "nominal_coverage": coverage,
                "empirical_coverage": empirical,
                "absolute_error": error,
                "passed": error is not None and error <= tolerance,
            })
    return {
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "maximum_absolute_coverage_error": tolerance,
        "checks": checks,
    }


def _run_diffusion(
    args: argparse.Namespace,
    forecaster: GNSSForecaster,
    bundle: dict[str, Any],
    train_loader: FastGPUTensorLoader,
    test_loader: FastGPUTensorLoader,
    device: torch.device,
    use_amp: bool,
    output: Path,
    test_real: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    if not args.enable_diffusion:
        return None
    schedule = DiffusionSchedule(
        steps=DIFFUSION_DEFAULTS["steps"],
        beta_start=DIFFUSION_DEFAULTS["beta_start"],
        beta_end=DIFFUSION_DEFAULTS["beta_end"],
        schedule_type="cosine",
        device=device,
    )
    denoiser = ConditionalDiffusionDenoiser(
        context_dim=forecaster.backbone.context_dim,
        d_model=args.d_model,
        output_dim=bundle["output_dim"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=DIFFUSION_DEFAULTS["learning_rate"],
        weight_decay=DIFFUSION_DEFAULTS["weight_decay"],
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    forecaster.eval()
    history = []
    for epoch in range(args.diffusion_epochs):
        denoiser.train()
        total, observations = 0.0, 0
        for x, y, sat, _, mask in train_loader:
            with torch.no_grad():
                mean, _, _, context = forecaster(x, sat)
            residual = torch.where(mask.bool(), y - mean, torch.zeros_like(y))
            timestep = torch.randint(0, schedule.steps, (len(y),), device=device)
            noisy_residual, noise = schedule.forward_sample(residual, timestep)
            noisy_residual = torch.where(mask.bool(), noisy_residual, torch.zeros_like(noisy_residual))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predicted_noise = denoiser(noisy_residual, context, timestep)
                loss = diffusion_mse_loss(predicted_noise, noise, mask)
            if use_amp:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach()) * len(y)
            observations += len(y)
        history.append(total / max(observations, 1))
        print(f"Diffusion epoch {epoch + 1:03d}/{args.diffusion_epochs:03d} loss={history[-1]:.6f}")

    torch.save(
        {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_state_dict": denoiser.state_dict(),
            "model_config": {
                "context_dim": forecaster.backbone.context_dim,
                "d_model": args.d_model,
                "output_dim": bundle["output_dim"],
                "num_layers": 2,
                "nhead": 4,
            },
            "schedule": {"steps": schedule.steps, "type": schedule.schedule_type},
            "target_cols": bundle["target_cols"],
            "target_scaler": scaler_to_state(bundle["target_scaler"]),
            "data_sha256": sha256_file(args.data),
            "history": history,
        },
        output / "gnss_diffusion_bundle.pt",
    )
    draws_scaled = []
    draw_count = max(2, int(args.diffusion_eval_samples))
    denoiser.eval()
    for _ in range(draw_count):
        batches = []
        for x, _, sat, _, _ in test_loader:
            with torch.no_grad():
                mean, _, _, context = forecaster(x, sat)
                generated = sample_ddim_forecast(
                    denoiser,
                    schedule,
                    context,
                    mean,
                    shape=mean.shape,
                    num_ddim_steps=args.ddim_steps,
                    device=device,
                )
            batches.append(generated.float().cpu().numpy())
        draws_scaled.append(np.concatenate(batches, axis=0))
    draws_scaled_array = np.stack(draws_scaled)
    target_dim = bundle["output_dim"]
    scaler = bundle["target_scaler"]
    draws_real = scaler.inverse_transform(draws_scaled_array.reshape(-1, target_dim)).reshape(draws_scaled_array.shape)
    draws_real = np.where(test_real["mask"][None, ...], draws_real, np.nan)
    sample_report = compute_sample_metrics(
        test_real["actual_real"], draws_real, bundle["target_cols"], valid_mask=test_real["mask"]
    )
    np.savez_compressed(output / "diffusion_test_samples.npz", samples=draws_real)
    if not args.skip_plots and len(draws_real):
        plot_diffusion_samples(
            test_real["actual_real"][0],
            [draw[0] for draw in draws_real[:10]],
            str(output / "05_diffusion_samples.png"),
        )
    return {"training_loss": history, "sample_metrics": sample_report}


def run_training(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(args.seed, deterministic=not args.nondeterministic)
    device = _resolve_device(args.device)
    use_amp = device.type == "cuda"
    print(f"Compute device: {device}; deterministic={not args.nondeterministic}")

    bundle = prepare_pytorch_datasets(
        args.data,
        input_window=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader = _gpu_loader(bundle, "train", device, args.batch_size, True)
    val_loader = _gpu_loader(bundle, "val", device, args.batch_size, False)
    test_loader = _gpu_loader(bundle, "test", device, args.batch_size, False)

    model = GNSSForecaster(**_model_config(args, bundle)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=TRANSFORMER_DEFAULTS["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=TRANSFORMER_DEFAULTS["lr_patience"]
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"loss": [], "val_loss": [], "scale_mean": []}
    best_loss, best_epoch, best_state, best_optimizer_state, patience = (
        float("inf"), -1, None, None, 0
    )
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, amp_scaler, use_amp, args)
        val_loss, scale_mean = validate_epoch(model, val_loader, use_amp, args)
        scheduler.step(val_loss)
        history["loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["scale_mean"].append(scale_mean)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} train={train_loss:.6f} "
            f"val={val_loss:.6f} scale={scale_mean:.4f}"
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

    metadata = _artifact_metadata(args, bundle)
    metadata.update({"best_epoch": best_epoch, "best_validation_loss": best_loss})
    torch.save(
        {
            **metadata,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": best_optimizer_state,
            "history": history,
        },
        output / "gnss_hybrid_forecaster_bundle.pt",
    )
    write_json(output / "artifact_manifest.json", metadata)

    val_real = _inverse_predictions(_collect_predictions(model, val_loader, use_amp), bundle["target_scaler"])
    test_real = _inverse_predictions(_collect_predictions(model, test_loader, use_amp), bundle["target_scaler"])
    satellite_ids = np.asarray(bundle["SATELLITE_IDS_test"], dtype=str)
    point_report = evaluate_forecasts(
        test_real["actual_real"],
        test_real["mean_real"],
        bundle["target_cols"],
        satellite_ids=satellite_ids,
        valid_mask=test_real["mask"],
    )
    probabilistic_report = compute_probabilistic_metrics(
        test_real["actual_real"],
        test_real["mean_real"],
        test_real["scale_real"],
        bundle["target_cols"],
        valid_mask=test_real["mask"],
        distribution=args.distribution,
        df=args.student_t_df,
    )
    calibration = fit_scaled_conformal(
        val_real["actual_real"],
        val_real["mean_real"],
        val_real["scale_real"],
        val_real["mask"],
    )
    conformal_report = evaluate_conformal_intervals(
        test_real["actual_real"],
        test_real["mean_real"],
        test_real["scale_real"],
        calibration,
        bundle["target_cols"],
        test_real["mask"],
    )
    calibration_gate = _calibration_gate(conformal_report)

    from src.baselines import generate_baseline_forecasts

    target_indices = np.asarray(bundle["target_feature_indices"], dtype=np.int64)
    feature_scaler = bundle["feature_scaler"]
    history_scaled = bundle["X_test"][:, :, target_indices]
    history_real = (
        history_scaled * np.asarray(feature_scaler.scale_)[target_indices][None, None, :]
        + np.asarray(feature_scaler.mean_)[target_indices][None, None, :]
    )
    baseline_predictions = generate_baseline_forecasts(
        history_real,
        horizon=FORECAST_HORIZON,
        season_length=SEQ_LEN,
    )
    baseline_reports, promotion = {}, {}
    for name, real_prediction in baseline_predictions.items():
        real_prediction = np.where(test_real["mask"], real_prediction, np.nan)
        report = evaluate_forecasts(
            test_real["actual_real"],
            real_prediction,
            bundle["target_cols"],
            satellite_ids=satellite_ids,
            valid_mask=test_real["mask"],
        )
        baseline_reports[name] = report
        promotion[name] = compare_candidate_to_baseline(
            point_report, report, _promotion_rules(bundle["target_cols"]), require_all=True
        )

    quality_report = audit_csv(args.data)
    evaluated_satellites = sorted(set(satellite_ids.tolist()))
    coverage = {
        "training_vocabulary_satellites": len(bundle["satellite_classes"]),
        "test_satellites_evaluated": len(evaluated_satellites),
        "evaluated_satellite_ids": evaluated_satellites,
        "test_samples": int(len(satellite_ids)),
        "valid_target_fraction": float(test_real["mask"].mean()),
    }
    diffusion_report = _run_diffusion(
        args, model, bundle, train_loader, test_loader, device, use_amp, output, test_real
    )
    final_report = {
        "model": point_report,
        "probabilistic": probabilistic_report,
        "conformal": {"calibration": calibration, "test": conformal_report},
        "calibration_promotion_gate": calibration_gate,
        "baselines": baseline_reports,
        "promotion_against_each_baseline": promotion,
        "promotion_eligible": bool(
            quality_report["passed"]
            and calibration_gate["passed"]
            and promotion
            and all(item["passed"] for item in promotion.values())
        ),
        "data_quality": quality_report,
        "coverage": coverage,
        "diffusion": diffusion_report,
    }
    write_json(output / "evaluation_report.json", final_report)
    np.savez_compressed(
        output / "test_predictions.npz",
        actual=test_real["actual_real"],
        mean=test_real["mean_real"],
        scale=test_real["scale_real"],
        valid_mask=test_real["mask"],
        satellite_ids=satellite_ids,
        label_timestamps=bundle["LABEL_TIMESTAMPS_test"],
    )

    if not args.skip_plots:
        plot_training_history(history, str(output / "01_transformer_training_history.png"))
        legacy_horizons = compute_tensor_horizon_metrics(
            test_real["actual_real"], test_real["mean_real"], bundle["target_cols"], HORIZON_MAP
        )
        plot_multihorizon_heatmap(
            legacy_horizons, bundle["target_cols"], str(output / "02_multihorizon_mae_heatmap.png")
        )
        plot_probabilistic_uncertainty(
            test_real["actual_real"][0],
            test_real["mean_real"][0],
            test_real["scale_real"][0],
            str(output / "03_probabilistic_uncertainty.png"),
        )
        plot_frequency_spectrum(
            test_real["actual_real"][0],
            test_real["mean_real"][0],
            str(output / "04_frequency_spectrum.png"),
        )
    print(f"Artifacts written to {output.resolve()}")
    print(f"Promotion eligible: {final_report['promotion_eligible']}")
    return final_report


if __name__ == "__main__":
    run_training()

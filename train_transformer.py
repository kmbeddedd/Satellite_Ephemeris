"""
CLI Entrypoint for Training & Evaluating the PyTorch Deep Transformer & Diffusion Pipeline
Optimized for GPU acceleration (CUDA), mixed precision, and multi-horizon validation.
"""

import argparse
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUTPUT_DIR,
    SEQ_LEN,
    FORECAST_HORIZON,
    TARGET_COLS_4,
    HORIZON_MAP,
    TRANSFORMER_DEFAULTS,
    DIFFUSION_DEFAULTS,
    DEFAULT_SEED
)
from src.data import prepare_pytorch_datasets
from src.models.pytorch_transformer import GNSSForecaster
from src.models.pytorch_diffusion import (
    DiffusionSchedule,
    ConditionalDiffusionDenoiser,
    sample_diffusion_forecast
)
from src.models.losses import composite_transformer_loss, diffusion_mse_loss
from src.evaluate import (
    compute_tensor_horizon_metrics,
    save_metrics_summary
)
from src.visualize import (
    plot_training_history,
    plot_multihorizon_heatmap,
    plot_probabilistic_uncertainty,
    plot_frequency_spectrum,
    plot_diffusion_samples
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PyTorch Deep Transformer & Diffusion GNSS Forecaster (GPU & CPU)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--output", default="./transformer_results", help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=TRANSFORMER_DEFAULTS["epochs"], help="Transformer training epochs")
    parser.add_argument("--diffusion-epochs", type=int, default=DIFFUSION_DEFAULTS["epochs"], help="Diffusion training epochs")
    parser.add_argument("--batch-size", type=int, default=TRANSFORMER_DEFAULTS["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=TRANSFORMER_DEFAULTS["learning_rate"], help="Transformer learning rate")
    parser.add_argument("--d-model", type=int, default=TRANSFORMER_DEFAULTS["d_model"], help="Transformer latent dimension")
    parser.add_argument("--nhead", type=int, default=TRANSFORMER_DEFAULTS["nhead"], help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=TRANSFORMER_DEFAULTS["num_layers"], help="Transformer encoder layers")
    parser.add_argument("--enable-diffusion", action="store_true", help="Enable training of conditional diffusion model")
    parser.add_argument("--device", default="auto", help="Target device: 'cuda', 'cuda:0', 'cpu', or 'auto'")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_epoch_loss = 0.0
    for x, y, sat, spike_targets in tqdm(loader, desc="Train Epoch", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sat = sat.to(device, non_blocking=True)
        spike_targets = spike_targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        mu, sigma, spike_probs, _ = model(x, sat)
        loss = composite_transformer_loss(mu, sigma, spike_probs, y, spike_targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_epoch_loss += loss.item()

    return total_epoch_loss / len(loader)


@torch.no_grad()
def validate_epoch(model, loader, device):
    model.eval()
    total_loss, sigmas, spikes = 0.0, [], []

    for x, y, sat, spike_targets in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sat = sat.to(device, non_blocking=True)
        spike_targets = spike_targets.to(device, non_blocking=True)

        mu, sigma, spike_probs, _ = model(x, sat)
        loss = composite_transformer_loss(mu, sigma, spike_probs, y, spike_targets)

        total_loss += loss.item()
        sigmas.append(sigma.mean().item())
        spikes.append(spike_probs.mean().item())

    return total_loss / len(loader), np.mean(sigmas), np.mean(spikes)


def train_diffusion_epoch(forecaster, diffusion_model, schedule, loader, optimizer, device):
    diffusion_model.train()
    total_epoch_loss = 0.0

    for x, y, sat, _ in tqdm(loader, desc="Diffusion Train Epoch", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sat = sat.to(device, non_blocking=True)
        batch_size = y.shape[0]

        with torch.no_grad():
            mu, _, _, context = forecaster(x, sat)

        t = torch.randint(0, schedule.steps, (batch_size,), device=device)
        residual = y - mu
        noisy_residual, true_noise = schedule.forward_sample(residual, t)

        optimizer.zero_grad()
        predicted_noise = diffusion_model(noisy_residual, context, t)
        loss = diffusion_mse_loss(predicted_noise, true_noise)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_epoch_loss += loss.item()

    return total_epoch_loss / len(loader)


def run_training():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = args.device

    device = torch.device(device_str)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"  Target Compute Device: GPU -> {gpu_name} ({total_vram:.2f} GB VRAM)")
    else:
        print("  Target Compute Device: CPU")

    # 1. Prepare Datasets & Scalers
    data_bundle = prepare_pytorch_datasets(
        data_path=args.data,
        input_window=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        batch_size=args.batch_size
    )

    train_loader = data_bundle["train_loader"]
    val_loader = data_bundle["val_loader"]
    test_loader = data_bundle["test_loader"]
    target_scaler = data_bundle["target_scaler"]

    # 2. Build Transformer Forecaster
    model = GNSSForecaster(
        num_features=data_bundle["num_features"],
        num_satellites=data_bundle["num_satellites"],
        d_model=args.d_model,
        forecast_horizon=FORECAST_HORIZON,
        output_dim=data_bundle["output_dim"],
        nhead=args.nhead,
        num_layers=args.num_layers
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=TRANSFORMER_DEFAULTS["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=TRANSFORMER_DEFAULTS["lr_patience"])

    train_losses, val_losses, sigma_means, spike_means = [], [], [], []

    print(f"\nTraining GNSS Transformer Forecaster ({args.epochs} epochs on {device})...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, sigma_mean, spike_mean = validate_epoch(model, val_loader, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        sigma_means.append(sigma_mean)
        spike_means.append(spike_mean)

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Sigma: {sigma_mean:.4f} | Spike: {spike_mean:.4f}")

    # Save Model Weights
    transformer_ckpt = os.path.join(args.output, "gnss_transformer.pt")
    torch.save(model.state_dict(), transformer_ckpt)
    print(f"\nTransformer Checkpoint Saved -> {transformer_ckpt}")

    # 3. Optional Diffusion Training
    diffusion_model = None
    schedule = None
    if args.enable_diffusion:
        print(f"\nTraining Conditional Diffusion Denoiser ({args.diffusion_epochs} epochs on {device})...")
        schedule = DiffusionSchedule(
            steps=DIFFUSION_DEFAULTS["steps"],
            beta_start=DIFFUSION_DEFAULTS["beta_start"],
            beta_end=DIFFUSION_DEFAULTS["beta_end"],
            device=device
        )
        diffusion_model = ConditionalDiffusionDenoiser(
            d_model=args.d_model,
            output_dim=data_bundle["output_dim"]
        ).to(device)

        diff_opt = torch.optim.AdamW(diffusion_model.parameters(), lr=DIFFUSION_DEFAULTS["learning_rate"], weight_decay=DIFFUSION_DEFAULTS["weight_decay"])

        for epoch in range(args.diffusion_epochs):
            diff_loss = train_diffusion_epoch(model, diffusion_model, schedule, train_loader, diff_opt, device)
            if (epoch + 1) % 10 == 0 or epoch == args.diffusion_epochs - 1:
                print(f"Diffusion Epoch {epoch+1:02d}/{args.diffusion_epochs:02d} | Loss: {diff_loss:.6f}")

        diffusion_ckpt = os.path.join(args.output, "gnss_diffusion.pt")
        torch.save(diffusion_model.state_dict(), diffusion_ckpt)
        print(f"Diffusion Checkpoint Saved -> {diffusion_ckpt}")

    # 4. Evaluation on Test Set
    print("\nRunning Evaluation on Test Set...")
    model.eval()
    mu_list, sigma_list, target_list = [], [], []

    with torch.no_grad():
        for x, y, sat, _ in test_loader:
            x = x.to(device, non_blocking=True)
            sat = sat.to(device, non_blocking=True)
            mu, sigma, _, _ = model(x, sat)

            mu_list.append(mu.cpu().numpy())
            sigma_list.append(sigma.cpu().numpy())
            target_list.append(y.numpy())

    mu = np.concatenate(mu_list, axis=0)
    sigma = np.concatenate(sigma_list, axis=0)
    targets = np.concatenate(target_list, axis=0)

    # Inverse Scale
    mu_real = target_scaler.inverse_transform(mu.reshape(-1, data_bundle["output_dim"])).reshape(mu.shape)
    target_real = target_scaler.inverse_transform(targets.reshape(-1, data_bundle["output_dim"])).reshape(targets.shape)
    sigma_real = sigma * target_scaler.scale_

    # Multi-Horizon Evaluation Metrics
    horizon_metrics = compute_tensor_horizon_metrics(target_real, mu_real, TARGET_COLS_4, HORIZON_MAP)

    aggregate = {
        col: {
            "Mean_MAE": float(np.mean(np.abs(mu_real[:, :, i] - target_real[:, :, i]))),
            "Mean_RMSE": float(np.sqrt(np.mean((mu_real[:, :, i] - target_real[:, :, i]) ** 2)))
        }
        for i, col in enumerate(TARGET_COLS_4)
    }

    save_metrics_summary(
        filepath=os.path.join(args.output, "transformer_metrics_summary.json"),
        model_name="Deep Multi-Task GNSS Transformer",
        aggregate=aggregate,
        horizon_results=horizon_metrics
    )

    # 5. Diagnostic Plots
    print("\nGenerating Diagnostic Visualizations...")
    plot_training_history(
        {"loss": train_losses, "val_loss": val_losses, "sigma_mean": sigma_means, "spike_mean": spike_means},
        os.path.join(args.output, "01_transformer_training_history.png"),
        title="Transformer Training History"
    )
    plot_multihorizon_heatmap(horizon_metrics, TARGET_COLS_4, os.path.join(args.output, "02_multihorizon_mae_heatmap.png"))
    plot_probabilistic_uncertainty(target_real[0], mu_real[0], sigma_real[0], os.path.join(args.output, "03_probabilistic_uncertainty.png"))
    plot_frequency_spectrum(target_real[0], mu_real[0], os.path.join(args.output, "04_frequency_spectrum.png"))

    if diffusion_model is not None:
        print("Generating Multi-Sample Diffusion Rollouts...")
        sample_x, _, sample_sat, _ = next(iter(test_loader))
        with torch.no_grad():
            s_mu, _, _, s_context = model(sample_x[:1].to(device), sample_sat[:1].to(device))

        diff_samples = []
        for _ in range(10):
            sample_gen = sample_diffusion_forecast(
                diffusion_model, schedule, s_context, s_mu, shape=s_mu.shape, device=device
            ).cpu().numpy()
            gen_real = target_scaler.inverse_transform(sample_gen.reshape(-1, data_bundle["output_dim"])).reshape(sample_gen.shape)
            diff_samples.append(gen_real[0])

        plot_diffusion_samples(target_real[0], diff_samples, os.path.join(args.output, "05_diffusion_samples.png"))

    print(f"\nTransformer & Diffusion Pipeline Complete! Artifacts saved to: {args.output}")


if __name__ == "__main__":
    run_training()

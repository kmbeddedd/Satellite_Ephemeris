"""
CLI Entrypoint for Deep Multi-Task Hybrid Architecture (BiLSTM-GRU-MHSA + Heads + Diffusion)
Maximized for 100% GPU Compute Utilization:
- In-VRAM GPU Data Resident Caching (FastGPUTensorLoader)
- Automatic Mixed Precision (AMP / FP16 Tensor Cores)
- Time2Vec and Dynamic PRN Entity Embeddings (de = ceil(1.6 * gamma^0.52))
- Probabilistic Gaussian Parameter Regression Head with Gaussian NLL
- Supervised Binary Event Classifier with BCE
- 100-step Conditional DDPM Diffusion Denoiser
- Multi-Horizon Metrics & Diagnostic Plots
"""

import argparse
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import (
    DEFAULT_DATA_PATH,
    SEQ_LEN,
    FORECAST_HORIZON,
    TARGET_COLS_4,
    HORIZON_MAP,
    TRANSFORMER_DEFAULTS,
    DIFFUSION_DEFAULTS,
    DEFAULT_SEED
)
from src.data import prepare_pytorch_datasets, FastGPUTensorLoader
from src.models.pytorch_transformer import GNSSForecaster
from src.models.pytorch_diffusion import ConditionalDiffusionDenoiser, DiffusionSchedule, sample_diffusion_forecast
from src.models.losses import composite_transformer_loss, diffusion_mse_loss
from src.evaluate import compute_tensor_horizon_metrics, save_metrics_summary
from src.visualize import (
    plot_training_history,
    plot_multihorizon_heatmap,
    plot_probabilistic_uncertainty,
    plot_frequency_spectrum,
    plot_diffusion_samples
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep Hybrid Sequence Forecaster with DDPM (Max GPU)")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument("--output", default="./transformer_results", help="Directory to save artifacts")
    parser.add_argument("--epochs", type=int, default=25, help="Transformer training epochs")
    parser.add_argument("--diffusion-epochs", type=int, default=20, help="Diffusion training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (larger saturates GPU cores)")
    parser.add_argument("--lr", type=float, default=2.0e-3, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--bilstm-units", type=int, default=48, help="BiLSTM hidden units")
    parser.add_argument("--gru-units", type=int, default=48, help="GRU hidden units")
    parser.add_argument("--nhead", type=int, default=4, help="Multi-Head Attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer encoder layers")
    parser.add_argument("--enable-diffusion", action="store_true", help="Train conditional DDPM module")
    parser.add_argument("--device", default="auto", help="Device ('cuda', 'cuda:0', 'cpu', or 'auto')")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scaler_amp, use_amp):
    model.train()
    total_loss = 0.0

    for x, y, sat, spikes in loader:
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            mu, sigma, spike_probs, _ = model(x, sat)
            loss = composite_transformer_loss(
                mu=mu,
                sigma=sigma,
                spike_probs=spike_probs,
                targets=y,
                spike_targets=spikes
            )

        if use_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, use_amp):
    model.eval()
    total_loss = 0.0
    sigmas = []
    spikes = []

    with torch.no_grad():
        for x, y, sat, spike_targets in loader:
            with torch.cuda.amp.autocast(enabled=use_amp):
                mu, sigma, spike_probs, _ = model(x, sat)
                loss = composite_transformer_loss(
                    mu=mu,
                    sigma=sigma,
                    spike_probs=spike_probs,
                    targets=y,
                    spike_targets=spike_targets
                )

            total_loss += loss.item()
            sigmas.append(sigma.mean().item())
            spikes.append(spike_probs.mean().item())

    return total_loss / len(loader), float(np.mean(sigmas)), float(np.mean(spikes))


def train_diffusion_epoch(forecaster, diffusion_model, schedule, loader, optimizer, scaler_amp, use_amp, device):
    diffusion_model.train()
    total_epoch_loss = 0.0

    for x, y, sat, _ in loader:
        batch_size = y.shape[0]

        with torch.no_grad():
            mu, _, _, context = forecaster(x, sat)

        t = torch.randint(0, schedule.steps, (batch_size,), device=device)
        residual = y - mu
        noisy_residual, true_noise = schedule.forward_sample(residual, t)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            predicted_noise = diffusion_model(noisy_residual, context, t)
            loss = diffusion_mse_loss(predicted_noise, true_noise)

        if use_amp:
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
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

    use_amp = (device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
        print(f"  Target Compute Device: GPU -> {gpu_name} ({total_vram:.2f} GB VRAM)")
        print(f"  Acceleration: In-VRAM GPU Direct Residency + Mixed Precision Tensor Cores")
    else:
        print("  Target Compute Device: CPU")

    # 1. Prepare Datasets & Scalers
    data_bundle = prepare_pytorch_datasets(
        data_path=args.data,
        input_window=SEQ_LEN,
        forecast_horizon=FORECAST_HORIZON,
        batch_size=args.batch_size
    )

    # In-VRAM GPU Loaders (Zero host-to-device bottlenecks)
    t_xtrain = torch.tensor(data_bundle["X_train"], dtype=torch.float32, device=device)
    t_ytrain = torch.tensor(data_bundle["Y_train"], dtype=torch.float32, device=device)
    t_sattrain = torch.tensor(data_bundle["SAT_train"], dtype=torch.long, device=device)
    t_spikrain = torch.tensor(data_bundle["SPIKE_train"], dtype=torch.float32, device=device)

    t_xval = torch.tensor(data_bundle["X_val"], dtype=torch.float32, device=device)
    t_yval = torch.tensor(data_bundle["Y_val"], dtype=torch.float32, device=device)
    t_satval = torch.tensor(data_bundle["SAT_val"], dtype=torch.long, device=device)
    t_spikval = torch.tensor(data_bundle["SPIKE_val"], dtype=torch.float32, device=device)

    t_xtest = torch.tensor(data_bundle["X_test"], dtype=torch.float32, device=device)
    t_ytest = torch.tensor(data_bundle["Y_test"], dtype=torch.float32, device=device)
    t_sattest = torch.tensor(data_bundle["SAT_test"], dtype=torch.long, device=device)
    t_spiktest = torch.tensor(data_bundle["SPIKE_test"], dtype=torch.float32, device=device)

    train_loader = FastGPUTensorLoader((t_xtrain, t_ytrain, t_sattrain, t_spikrain), batch_size=args.batch_size, shuffle=True, device=device)
    val_loader = FastGPUTensorLoader((t_xval, t_yval, t_satval, t_spikval), batch_size=args.batch_size, shuffle=False, device=device)
    test_loader = FastGPUTensorLoader((t_xtest, t_ytest, t_sattest, t_spiktest), batch_size=args.batch_size, shuffle=False, device=device)

    target_scaler = data_bundle["target_scaler"]

    # 2. Build Deep Multi-Task Hybrid Forecaster
    model = GNSSForecaster(
        num_features=data_bundle["num_features"],
        num_satellites=data_bundle["num_satellites"],
        d_model=args.d_model,
        bilstm_units=args.bilstm_units,
        gru_units=args.gru_units,
        nhead=args.nhead,
        forecast_horizon=FORECAST_HORIZON,
        output_dim=data_bundle["output_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=TRANSFORMER_DEFAULTS["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=TRANSFORMER_DEFAULTS["lr_patience"])
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_losses, val_losses, sigma_means, spike_means = [], [], [], []

    print(f"\nTraining Hybrid Sequence Forecaster ({args.epochs} epochs, batch={args.batch_size})...")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler_amp, use_amp)
        val_loss, sigma_mean, spike_mean = validate_epoch(model, val_loader, use_amp)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        sigma_means.append(sigma_mean)
        spike_means.append(spike_mean)

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Sigma: {sigma_mean:.4f} | Spike: {spike_mean:.4f}")

    # Save Model Weights
    transformer_ckpt = os.path.join(args.output, "gnss_hybrid_forecaster.pt")
    torch.save(model.state_dict(), transformer_ckpt)
    print(f"\nHybrid Forecaster Checkpoint Saved -> {transformer_ckpt}")

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
            context_dim=model.backbone.context_dim,
            d_model=args.d_model,
            output_dim=data_bundle["output_dim"]
        ).to(device)

        diff_opt = torch.optim.AdamW(diffusion_model.parameters(), lr=DIFFUSION_DEFAULTS["learning_rate"], weight_decay=DIFFUSION_DEFAULTS["weight_decay"])
        diff_scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

        for epoch in range(args.diffusion_epochs):
            diff_loss = train_diffusion_epoch(model, diffusion_model, schedule, train_loader, diff_opt, diff_scaler_amp, use_amp, device)
            if (epoch + 1) % 10 == 0 or epoch == args.diffusion_epochs - 1:
                print(f"Diffusion Epoch {epoch+1:02d}/{args.diffusion_epochs:02d} | Loss: {diff_loss:.6f}")

        diffusion_ckpt = os.path.join(args.output, "gnss_diffusion.pt")
        torch.save(diffusion_model.state_dict(), diffusion_ckpt)
        print(f"Diffusion Checkpoint Saved -> {diffusion_ckpt}")

    # 4. Evaluation on Test Set
    print("\nRunning Evaluation on Out-of-Sample Test Set...")
    model.eval()
    mu_list, sigma_list, target_list = [], [], []

    with torch.no_grad():
        for x, y, sat, _ in test_loader:
            with torch.cuda.amp.autocast(enabled=use_amp):
                mu, sigma, _, _ = model(x, sat)

            mu_list.append(mu.float().cpu().numpy())
            sigma_list.append(sigma.float().cpu().numpy())
            target_list.append(y.float().cpu().numpy())

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
        model_name="Deep Multi-Task Hybrid Forecaster (BiLSTM-GRU-MHSA + DDPM)",
        aggregate=aggregate,
        horizon_results=horizon_metrics
    )

    # 5. Diagnostic Plots
    print("\nGenerating Publication Diagnostic Visualizations...")
    plot_training_history(
        {"loss": train_losses, "val_loss": val_losses, "sigma_mean": sigma_means, "spike_mean": spike_means},
        os.path.join(args.output, "01_transformer_training_history.png"),
        title="Hybrid Sequence Model Training History"
    )
    plot_multihorizon_heatmap(horizon_metrics, TARGET_COLS_4, os.path.join(args.output, "02_multihorizon_mae_heatmap.png"))
    plot_probabilistic_uncertainty(target_real[0], mu_real[0], sigma_real[0], os.path.join(args.output, "03_probabilistic_uncertainty.png"))
    plot_frequency_spectrum(target_real[0], mu_real[0], os.path.join(args.output, "04_frequency_spectrum.png"))

    if diffusion_model is not None:
        print("Generating Multi-Sample Diffusion Rollouts...")
        sample_x = t_xtest[:1]
        sample_sat = t_sattest[:1]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                s_mu, _, _, s_context = model(sample_x, sample_sat)

        diff_samples = []
        for _ in range(10):
            sample_gen = sample_diffusion_forecast(
                diffusion_model, schedule, s_context, s_mu, shape=s_mu.shape, device=device
            ).float().cpu().numpy()
            gen_real = target_scaler.inverse_transform(sample_gen.reshape(-1, data_bundle["output_dim"])).reshape(sample_gen.shape)
            diff_samples.append(gen_real[0])

        plot_diffusion_samples(target_real[0], diff_samples, os.path.join(args.output, "05_diffusion_samples.png"))

    print(f"\nHybrid Pipeline Complete! All artifacts saved to: {args.output}")


if __name__ == "__main__":
    run_training()

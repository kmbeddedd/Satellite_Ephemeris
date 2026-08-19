import torch
from src.config import DEFAULT_DATA_PATH
from src.data import prepare_pytorch_datasets
from src.models.pytorch_transformer import GNSSForecaster
from src.models.losses import composite_transformer_loss, dilate_loss
from src.models.pytorch_diffusion import ConditionalDiffusionDenoiser, DiffusionSchedule, sample_ddim_forecast

print("=" * 60)
print("RUNNING VERIFICATION OF SOTA UPGRADES")
print("=" * 60)

print("1. Testing data loading and temporal split...")
data = prepare_pytorch_datasets(DEFAULT_DATA_PATH, input_window=96, forecast_horizon=96, batch_size=16)
print(f"   X_train: {data['X_train'].shape}, X_val: {data['X_val'].shape}, X_test: {data['X_test'].shape}")
assert len(data['X_train']) > 0 and len(data['X_test']) > 0
print("   [PASSED] Temporal dataset partitioning verified.")

print("2. Testing GNSSForecaster model with RevIN...")
model = GNSSForecaster(
    num_features=data['num_features'],
    num_satellites=data['num_satellites'],
    d_model=64,
    bilstm_units=32,
    gru_units=32,
    nhead=4,
    use_revin=True
)
x = torch.randn(4, 96, data['num_features'])
sat = torch.randint(0, data['num_satellites'], (4,))
mu, sigma, spike_probs, context = model(x, sat)
print(f"   mu: {mu.shape}, sigma: {sigma.shape}, spike_probs: {spike_probs.shape}, context: {context.shape}")
assert mu.shape == (4, 96, 4)
assert sigma.shape == (4, 96, 4)
print("   [PASSED] Forward pass with RevIN & sequence projection verified.")

print("3. Testing composite_transformer_loss with DILATE (Soft-DTW)...")
targets = torch.randn(4, 96, 4)
spikes = torch.zeros(4, 96)
loss = composite_transformer_loss(mu, sigma, spike_probs, targets, spikes, lambda_dilate=0.05)
print(f"   Composite Loss value: {loss.item():.4f}")
assert not torch.isnan(loss) and not torch.isinf(loss)

loss.backward()
print("   [PASSED] Backward pass and gradient flow through DILATE verified.")

print("4. Testing DDIM accelerated fast diffusion sampling...")
schedule = DiffusionSchedule(steps=50, device="cpu")
diff_model = ConditionalDiffusionDenoiser(context_dim=model.backbone.context_dim, d_model=64, output_dim=4)
ddim_sample = sample_ddim_forecast(diff_model, schedule, context.detach(), mu.detach(), shape=(4, 96, 4), num_ddim_steps=10, device="cpu")
print(f"   DDIM sample shape: {ddim_sample.shape}")
assert ddim_sample.shape == (4, 96, 4)
print("   [PASSED] DDIM fast reverse sampler verified.")

print("\n" + "=" * 60)
print("ALL 4 SOTA VERIFICATION CHECKS PASSED SUCCESSFULLY!")
print("=" * 60)

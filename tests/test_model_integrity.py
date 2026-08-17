import torch

from src.models.losses import composite_transformer_loss, gaussian_nll_loss
from src.models.pytorch_diffusion import DiffusionSchedule, sample_ddim_forecast
from src.models.pytorch_transformer import GNSSForecaster, RevIN


def test_masked_loss_ignores_missing_target_values():
    mu = torch.zeros(1, 3, 2)
    sigma = torch.ones_like(mu)
    target_a = torch.zeros_like(mu)
    target_b = target_a.clone()
    target_b[0, 1, 1] = 1_000_000.0
    mask = torch.ones_like(mu)
    mask[0, 1, 1] = 0.0

    assert torch.allclose(
        gaussian_nll_loss(mu, sigma, target_a, mask),
        gaussian_nll_loss(mu, sigma, target_b, mask),
    )


def test_composite_student_t_loss_is_finite_with_mask():
    mu = torch.randn(2, 6, 4, requires_grad=True)
    sigma = torch.rand_like(mu).clamp_min(0.1)
    targets = torch.randn_like(mu)
    mask = torch.ones_like(mu)
    mask[:, 2, 3] = 0.0
    loss = composite_transformer_loss(
        mu,
        sigma,
        None,
        targets,
        None,
        target_mask=mask,
        distribution="student_t",
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(mu.grad).all()


def test_revin_sigma_undoes_affine_scale():
    layer = RevIN(num_features=2, affine=True)
    x = torch.tensor([[[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]])
    layer(x, mode="norm")
    with torch.no_grad():
        layer.affine_weight.copy_(torch.tensor([2.0, -4.0]))
    sigma = torch.ones(1, 1, 2)
    actual = layer(sigma, mode="denorm_sigma")
    expected = layer.stdev / torch.tensor([2.0, 4.0])
    assert torch.allclose(actual, expected)


def test_attention_depth_and_disabled_event_api():
    model = GNSSForecaster(
        num_features=6,
        num_satellites=3,
        d_model=8,
        bilstm_units=4,
        gru_units=8,
        nhead=2,
        num_layers=3,
        seq_len=8,
        forecast_horizon=4,
        use_revin=False,
        enable_event_head=False,
    )
    assert len(model.backbone.attention_layers) == 2
    mu, sigma, event_logits, _ = model(torch.randn(2, 8, 6), torch.tensor([0, 2]))
    assert mu.shape == sigma.shape == (2, 4, 4)
    assert event_logits.shape == (2, 4)
    assert torch.count_nonzero(event_logits) == 0


class _RecordingZeroDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.max_abs_input = 0.0
        self.first_abs_input = None

    def forward(self, residual, context, timestep):
        if self.first_abs_input is None:
            self.first_abs_input = float(residual.abs().max())
        self.max_abs_input = max(self.max_abs_input, float(residual.abs().max()))
        return torch.zeros_like(residual)


def test_diffusion_schedule_reaches_noise_and_samples_residual_space():
    torch.manual_seed(7)
    schedule = DiffusionSchedule(steps=20, device="cpu")
    assert float(schedule.alpha_bars[-1]) < 1e-3
    denoiser = _RecordingZeroDenoiser()
    mu = torch.full((2, 4, 4), 100.0)
    output = sample_ddim_forecast(
        denoiser,
        schedule,
        torch.zeros(2, 5),
        mu,
        shape=mu.shape,
        num_ddim_steps=5,
        device="cpu",
    )
    assert output.shape == mu.shape
    assert denoiser.first_abs_input < 50.0  # The denoiser was not fed mu + residual.

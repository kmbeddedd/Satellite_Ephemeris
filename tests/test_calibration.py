import numpy as np

from src.calibration import conformal_interval, evaluate_conformal_intervals, fit_scaled_conformal


def test_scaled_conformal_handles_mask_and_produces_ordered_intervals():
    rng = np.random.default_rng(4)
    actual = rng.normal(size=(40, 3, 2))
    mean = np.zeros_like(actual)
    scale = np.ones_like(actual)
    mask = np.ones_like(actual, dtype=bool)
    mask[:10, 1, 1] = False
    calibration = fit_scaled_conformal(actual, mean, scale, mask, coverages=(0.9,))
    lower, upper = conformal_interval(mean, scale, calibration, 0.9)
    assert np.all(lower <= upper)
    report = evaluate_conformal_intervals(
        actual, mean, scale, calibration, ["Error_X", "Error_Clock"], mask
    )
    assert report["per_target"]["Error_Clock"]["0.9"]["count"] == int(mask[..., 1].sum())


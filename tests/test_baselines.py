"""Tests for executable simple forecast baselines and reporting CLI."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_baselines import main as baseline_cli_main
from src.baselines import (
    drift_forecast,
    evaluate_baselines,
    generate_baseline_forecasts,
    persistence_forecast,
    seasonal_forecast,
    zero_forecast,
)


TARGETS = ["Error_X", "Error_Y", "Error_Z", "Error_Clock"]


class BaselineForecastTests(unittest.TestCase):
    def test_zero_and_persistence_preserve_unbatched_shape(self):
        history = np.array([[1.0, 10.0], [2.0, 20.0]])
        np.testing.assert_array_equal(zero_forecast(history, 3), np.zeros((3, 2)))
        np.testing.assert_array_equal(
            persistence_forecast(history, 3),
            np.array([[2.0, 20.0], [2.0, 20.0], [2.0, 20.0]]),
        )

    def test_seasonal_cycles_final_complete_season(self):
        history = np.arange(1.0, 5.0)[:, None]
        forecast = seasonal_forecast(history, horizon=5, season_length=2)
        np.testing.assert_array_equal(forecast[:, 0], [3.0, 4.0, 3.0, 4.0, 3.0])

    def test_drift_extrapolates_first_to_last_line(self):
        forecast = drift_forecast(np.array([1.0, 3.0]), horizon=3)
        np.testing.assert_array_equal(forecast, [5.0, 7.0, 9.0])

    def test_batched_generation_and_unknown_name(self):
        history = np.arange(16.0).reshape(2, 4, 2)
        forecasts = generate_baseline_forecasts(
            history, 3, baselines=["zero", "persistence", "seasonal", "drift"], season_length=2
        )
        self.assertEqual(set(forecasts), {"zero", "persistence", "seasonal", "drift"})
        self.assertTrue(all(forecast.shape == (2, 3, 2) for forecast in forecasts.values()))
        with self.assertRaisesRegex(ValueError, "unknown baselines"):
            generate_baseline_forecasts(history, 2, baselines=["oracle"])

    def test_evaluate_baselines_honours_label_mask(self):
        history = np.ones((1, 2, 4))
        actual = np.ones((1, 2, 4))
        mask = np.ones_like(actual, dtype=bool)
        mask[0, 0, 3] = False
        reports = evaluate_baselines(
            history,
            actual,
            TARGETS,
            baselines=["zero", "persistence"],
            horizons={"two": 2},
            satellite_ids=["G01"],
            valid_mask=mask,
        )
        self.assertEqual(
            reports["zero"]["all_forecast_points"]["per_target"]["Error_Clock"]["count"],
            1,
        )
        self.assertEqual(
            reports["persistence"]["all_forecast_points"]["per_target"]["Error_X"]["mae"],
            0.0,
        )


class BaselineCliTests(unittest.TestCase):
    def test_failed_promotion_is_reported_without_nonzero_exit_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timestamps = pd.date_range("2026-01-01", periods=6, freq="15min")
            frame = pd.DataFrame(
                {
                    "Timestamp": timestamps,
                    "Satellite_ID": "G01",
                    "Constellation": "G",
                    "Error_X": np.arange(6.0),
                    "Error_Y": 0.0,
                    "Error_Z": 0.0,
                    "Error_Clock": np.arange(6.0) * 1e-9,
                }
            )
            data_path = root / "data.csv"
            output_path = root / "report.json"
            prediction_path = root / "candidate.npy"
            frame.to_csv(data_path, index=False)
            # Candidate equals the zero baseline, so strict "must beat" gates fail.
            np.save(prediction_path, np.zeros((1, 2, 4)))
            exit_code = baseline_cli_main(
                [
                    "--data",
                    str(data_path),
                    "--horizon",
                    "2",
                    "--lookback",
                    "2",
                    "--baselines",
                    "zero",
                    "--candidate-predictions",
                    str(prediction_path),
                    "--output",
                    str(output_path),
                ]
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertFalse(report["promotion"]["passed"])


if __name__ == "__main__":
    unittest.main()

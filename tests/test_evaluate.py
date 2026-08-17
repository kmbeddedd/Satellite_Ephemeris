"""Regression tests for unit-aware deterministic/probabilistic evaluation."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t_distribution

from src.evaluate import (
    SPEED_OF_LIGHT_M_S,
    PromotionGateError,
    assert_candidate_beats_baseline,
    compare_candidate_to_baseline,
    compute_aggregate_metrics,
    compute_probabilistic_metrics,
    compute_sample_metrics,
    compute_skill_score,
    compute_tensor_horizon_metrics,
    evaluate_forecasts,
    save_metrics_summary,
)


TARGETS = ["Error_X", "Error_Y", "Error_Z", "Error_Clock"]


class DeterministicEvaluationTests(unittest.TestCase):
    def test_exact_lead_and_cumulative_window_are_both_explicit(self):
        actual = np.zeros((1, 2, 4))
        predicted = np.zeros_like(actual)
        predicted[0, :, 0] = [1.0, 3.0]
        predicted[0, :, 3] = [1e-9, 3e-9]

        report = evaluate_forecasts(
            actual, predicted, TARGETS, horizons={"lead 2": 2}
        )
        horizon = report["horizons"]["lead 2"]
        exact = horizon["exact_lead"]["per_target"]["Error_X"]
        cumulative = horizon["cumulative_window_1_to_lead"]["per_target"]["Error_X"]
        self.assertEqual(exact["mae"], 3.0)
        self.assertEqual(cumulative["mae"], 2.0)
        self.assertAlmostEqual(cumulative["rmse"], math.sqrt(5.0))
        self.assertEqual(cumulative["median_ae"], 2.0)
        self.assertAlmostEqual(cumulative["p90_ae"], 2.8)
        self.assertNotIn("overall", report["all_forecast_points"])

    def test_3d_orbit_error_is_derived_from_xyz_not_scalar_target(self):
        targets = ["Error_X", "Error_Y", "Error_Z", "3D_Orbit_Error", "Error_Clock"]
        actual = np.zeros((1, 1, 5))
        predicted = np.array([[[3.0, 4.0, 0.0, 9999.0, 0.0]]])
        report = evaluate_forecasts(actual, predicted, targets, horizons={"one": 1})
        block = report["all_forecast_points"]
        self.assertEqual(block["orbit_3d_vector_error"]["mae"], 5.0)
        self.assertNotIn("3D_Orbit_Error", block["per_target"])

        per_satellite, aggregate = compute_aggregate_metrics(
            {"G01": actual[0]}, {"G01": predicted[0]}, targets
        )
        self.assertEqual(per_satellite["G01"]["3D_Orbit_Error"]["MAE"], 5.0)
        self.assertEqual(aggregate["3D_Orbit_Error"]["Mean_MAE"], 5.0)

    def test_clock_is_reported_in_seconds_nanoseconds_and_range_metres(self):
        actual = np.zeros((1, 2, 4))
        predicted = np.zeros_like(actual)
        predicted[..., 3] = [[1e-9, 3e-9]]
        clock = evaluate_forecasts(
            actual, predicted, TARGETS, horizons={"two": 2}
        )["all_forecast_points"]["clock_error"]
        self.assertAlmostEqual(clock["seconds"]["mae"], 2e-9)
        self.assertAlmostEqual(clock["nanoseconds"]["mae"], 2.0)
        self.assertAlmostEqual(
            clock["range_equivalent_metres"]["mae"], 2e-9 * SPEED_OF_LIGHT_M_S
        )

    def test_satellite_constellation_slices_and_mask(self):
        actual = np.zeros((2, 2, 4))
        predicted = np.zeros_like(actual)
        predicted[0, :, 0] = 1.0
        predicted[1, :, 0] = 3.0
        valid_mask = np.ones_like(actual, dtype=bool)
        valid_mask[1, 0, 0] = False
        report = evaluate_forecasts(
            actual,
            predicted,
            TARGETS,
            horizons={"two": 2},
            satellite_ids=["G01", "R01"],
            valid_mask=valid_mask,
        )
        slices = report["slices"]
        self.assertEqual(set(slices["per_satellite"]), {"G01", "R01"})
        self.assertEqual(set(slices["per_constellation"]), {"G", "R"})
        r_x = slices["per_satellite"]["R01"]["all_forecast_points"]["per_target"][
            "Error_X"
        ]
        self.assertEqual(r_x["count"], 1)
        self.assertEqual(r_x["mae"], 3.0)

    def test_unavailable_horizon_is_not_silently_clamped(self):
        values = np.zeros((1, 2, 4))
        report = evaluate_forecasts(values, values, TARGETS, horizons={"too far": 3})
        self.assertFalse(report["horizons"]["too far"]["available"])
        self.assertIsNone(report["horizons"]["too far"]["exact_lead"])

    def test_tensor_compatibility_wrapper_does_not_mix_units(self):
        actual = np.zeros((1, 1, 4))
        predicted = np.ones_like(actual)
        metrics = compute_tensor_horizon_metrics(
            actual, predicted, TARGETS, horizons={"one": 1}
        )["one"]
        self.assertNotIn("Overall_MAE", metrics)
        self.assertIn("exact_lead", metrics)
        self.assertIn("cumulative_window_1_to_lead", metrics)

    def test_json_writer_replaces_non_finite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            save_metrics_summary(
                str(path),
                "test",
                {"Error_X": {"Mean_MAE": float("nan")}},
                {},
            )
            self.assertIn('"Mean_MAE": null', path.read_text(encoding="utf-8"))


class ProbabilisticEvaluationTests(unittest.TestCase):
    def test_gaussian_nll_coverage_and_mask(self):
        actual = np.zeros((1, 2, 1))
        mean = np.zeros_like(actual)
        sigma = np.ones_like(actual)
        mask = np.array([[[True], [False]]])
        metrics = compute_probabilistic_metrics(
            actual,
            mean,
            sigma,
            ["Error_X"],
            horizons={"two": 2},
            coverages=(0.9,),
            valid_mask=mask,
        )["all_forecast_points"]["per_target"]["Error_X"]
        self.assertEqual(metrics["count"], 1)
        self.assertAlmostEqual(metrics["gaussian_nll"], 0.5 * math.log(2 * math.pi))
        self.assertEqual(metrics["intervals"]["90%"]["empirical_coverage"], 1.0)

    def test_student_t_nll_and_intervals(self):
        actual = np.zeros((1, 1, 1))
        metrics = compute_probabilistic_metrics(
            actual,
            actual,
            np.ones_like(actual),
            ["Error_X"],
            horizons={"one": 1},
            coverages=(0.9,),
            distribution="student_t",
            df=3.0,
        )["all_forecast_points"]["per_target"]["Error_X"]
        self.assertAlmostEqual(
            metrics["student_t_nll"], -student_t_distribution.logpdf(0.0, df=3.0)
        )
        self.assertEqual(metrics["mean_degrees_of_freedom"], 3.0)
        self.assertGreater(metrics["intervals"]["90%"]["mean_width"], 0.0)

    def test_empirical_crps_and_xyz_energy_score(self):
        actual = np.zeros((1, 1, 3))
        samples = np.array(
            [
                [[[1.0, 0.0, 0.0]]],
                [[[-1.0, 0.0, 0.0]]],
            ]
        )
        report = compute_sample_metrics(
            actual,
            samples,
            ["Error_X", "Error_Y", "Error_Z"],
            horizons={"one": 1},
        )["all_forecast_points"]
        self.assertAlmostEqual(report["per_target"]["Error_X"]["crps"], 0.5)
        self.assertAlmostEqual(report["orbit_xyz_energy_score"]["energy_score"], 0.5)


class PromotionGateTests(unittest.TestCase):
    RULES = {"mae": {"direction": "lower", "min_relative_improvement": 0.05}}

    def test_skill_and_promotion_pass(self):
        self.assertAlmostEqual(compute_skill_score(8.0, 10.0), 0.2)
        report = compare_candidate_to_baseline({"mae": 8.0}, {"mae": 10.0}, self.RULES)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["comparisons"][0]["skill_score"], 0.2)

    def test_api_raises_when_candidate_does_not_beat_baseline(self):
        with self.assertRaises(PromotionGateError):
            assert_candidate_beats_baseline({"mae": 10.0}, {"mae": 10.0}, self.RULES)

    def test_missing_metric_fails_closed(self):
        report = compare_candidate_to_baseline({}, {"mae": 10.0}, self.RULES)
        self.assertFalse(report["passed"])
        self.assertEqual(report["total_rules"], 1)


if __name__ == "__main__":
    unittest.main()

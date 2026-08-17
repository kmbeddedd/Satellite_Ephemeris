"""Focused regression tests for the GNSS data contract and temporal splits."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_and_clean_data, prepare_pytorch_datasets


TARGETS = ["Error_X", "Error_Y", "Error_Z", "Error_Clock"]


def synthetic_frame(
    epochs: int = 80,
    *,
    sentinel_index: int | None = 10,
    missing_index: int | None = None,
) -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01", periods=epochs, freq="15min")
    index = np.arange(epochs, dtype=np.float64)
    frame = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Satellite_ID": "G01",
            "Constellation": "G",
            "Broadcast_X": 20_000_000.0 + 1000.0 * index,
            "Broadcast_Y": 15_000_000.0 - 400.0 * index,
            "Broadcast_Z": 10_000_000.0 + 200.0 * index,
            "Broadcast_Clock": 2e-4 + index * 1e-8,
            "Modelled_Clock": 2e-4 + index * 5e-9,
            "Error_X": 10.0 + 0.25 * index,
            "Error_Y": -20.0 + 0.5 * index,
            "Error_Z": np.sin(index / 5.0) * 3.0,
            "Error_Clock": index * 5e-9,
            "3D_Orbit_Error": 25.0 + index,
        }
    )
    if sentinel_index is not None:
        frame.loc[sentinel_index, "Modelled_Clock"] = 0.999999999999
        frame.loc[sentinel_index, "Error_Clock"] = (
            frame.loc[sentinel_index, "Broadcast_Clock"] - 0.999999999999
        )
    # A target-dependent outlier must remain a row, not destroy cadence.
    frame.loc[5, "3D_Orbit_Error"] = 100_000.0
    if missing_index is not None:
        frame = frame.drop(index=missing_index).reset_index(drop=True)
    return frame


class DataPipelineContractTests(unittest.TestCase):
    def write_csv(self, frame: pd.DataFrame, directory: str) -> Path:
        path = Path(directory) / "synthetic.csv"
        frame.to_csv(path, index=False)
        return path

    def prepare(self, path: Path):
        return prepare_pytorch_datasets(
            str(path),
            input_window=4,
            forecast_horizon=3,
            batch_size=8,
            train_end_date="2025-01-01 15:00:00",  # epoch 60
            seed=7,
        )

    def test_sp3_clock_sentinel_is_masked_without_removing_row(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            train, test, satellites = load_and_clean_data(
                str(path), train_end_date="2025-01-01 15:00:00"
            )

        combined = pd.concat([train, test], ignore_index=True)
        sentinel = combined[combined["SP3_Clock_Sentinel"]]
        self.assertEqual(len(combined), len(frame))
        self.assertEqual(satellites, ["G01"])
        self.assertEqual(len(sentinel), 1)
        self.assertFalse(bool(sentinel.iloc[0]["Error_Clock_valid"]))
        self.assertTrue(np.isnan(sentinel.iloc[0]["Error_Clock"]))
        self.assertEqual(int((combined["3D_Orbit_Error"] >= 50_000).sum()), 1)

    def test_targets_are_scaled_once_and_invalid_clock_never_trains(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)

        fit_end = pd.Timestamp(bundle["split_metadata"]["scaler_fit_end_exclusive"])
        expected_clock_mean = frame.loc[
            (frame["Timestamp"] < fit_end)
            & ~np.isclose(frame["Modelled_Clock"], 0.999999999999, atol=1e-9, rtol=0),
            "Error_Clock",
        ].mean()
        self.assertAlmostEqual(bundle["target_scaler"].mean_[3], expected_clock_mean)
        expected_x_mean = frame.loc[frame["Timestamp"] < fit_end, "Error_X"].mean()
        self.assertAlmostEqual(bundle["target_scaler"].mean_[0], expected_x_mean)
        self.assertAlmostEqual(bundle["feature_scaler"].mean_[0], expected_x_mean)
        self.assertEqual(bundle["target_feature_indices"], [0, 1, 2, 3])
        self.assertIn("Broadcast_VX", bundle["feature_cols"])
        self.assertIn("Broadcast_Phase_Sin", bundle["feature_cols"])
        json.dumps(bundle["data_quality_report"])
        json.dumps(bundle["split_metadata"])
        self.assertEqual(len(next(iter(bundle["train_loader"]))), 5)

        saw_invalid_clock = False
        lookup = frame.set_index(["Satellite_ID", "Timestamp"])
        for split in ("train", "val", "test"):
            restored = bundle["target_scaler"].inverse_transform(
                bundle[f"Y_{split}"].reshape(-1, 4)
            ).reshape(bundle[f"Y_{split}"].shape)
            masks = bundle[f"TARGET_MASK_{split}"].astype(bool)
            self.assertEqual(masks.shape, bundle[f"Y_{split}"].shape)
            self.assertTrue(np.isfinite(bundle[f"Y_{split}"]).all())
            for sample_index, satellite_id in enumerate(bundle[f"SATELLITE_IDS_{split}"]):
                for horizon_index, timestamp in enumerate(
                    bundle[f"LABEL_TIMESTAMPS_{split}"][sample_index]
                ):
                    raw = lookup.loc[(satellite_id, pd.Timestamp(timestamp)), TARGETS].to_numpy(float)
                    valid = masks[sample_index, horizon_index]
                    np.testing.assert_allclose(
                        restored[sample_index, horizon_index, valid],
                        raw[valid],
                        rtol=1e-5,
                        atol=1e-7,
                    )
                    if not valid[3]:
                        saw_invalid_clock = True
                        self.assertEqual(
                            bundle[f"Y_{split}"][sample_index, horizon_index, 3], 0.0
                        )
        self.assertTrue(saw_invalid_clock)

    def test_every_emitted_window_is_contiguous_even_when_source_has_gap(self):
        frame = synthetic_frame(sentinel_index=None, missing_index=25)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)

        self.assertEqual(bundle["data_quality_report"]["irregular_steps"], 1)
        self.assertGreater(
            bundle["data_quality_report"]["skipped_noncontiguous_windows"], 0
        )
        expected_ns = pd.Timedelta(minutes=15).value
        for split in ("train", "val", "test"):
            combined = np.concatenate(
                [
                    bundle[f"INPUT_TIMESTAMPS_{split}"],
                    bundle[f"LABEL_TIMESTAMPS_{split}"],
                ],
                axis=1,
            ).astype("datetime64[ns]").astype(np.int64)
            self.assertTrue(np.all(np.diff(combined, axis=1) == expected_ns))

    def test_train_validation_and_test_label_timestamps_are_disjoint(self):
        frame = synthetic_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            bundle = self.prepare(path)

        labels = {
            split: set(bundle[f"LABEL_TIMESTAMPS_{split}"].reshape(-1).tolist())
            for split in ("train", "val", "test")
        }
        self.assertTrue(labels["train"].isdisjoint(labels["val"]))
        self.assertTrue(labels["train"].isdisjoint(labels["test"]))
        self.assertTrue(labels["val"].isdisjoint(labels["test"]))
        self.assertGreater(bundle["split_metadata"]["purged_boundary_windows"], 0)

    def test_insufficient_history_has_actionable_error(self):
        frame = synthetic_frame(epochs=10, sentinel_index=None)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_csv(frame, directory)
            with self.assertRaisesRegex(ValueError, "Insufficient history"):
                prepare_pytorch_datasets(
                    str(path),
                    input_window=4,
                    forecast_horizon=3,
                    train_end_date="2025-01-01 01:30:00",
                )


if __name__ == "__main__":
    unittest.main()

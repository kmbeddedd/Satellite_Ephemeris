"""Read-only data-contract audit for GNSS training CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import sha256_file, write_json


def audit_csv(data_path: str, contract_path: str = "configs/data_contract.json") -> dict:
    with open(contract_path, encoding="utf-8") as handle:
        contract = json.load(handle)
    frame = pd.read_csv(data_path)
    required = contract["required_columns"]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        return {
            "data_path": data_path,
            "sha256": sha256_file(data_path),
            "passed": False,
            "critical_failures": [f"missing required columns: {missing_columns}"],
        }

    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")
    invalid_timestamps = int(frame["Timestamp"].isna().sum())
    duplicate_keys = int(frame.duplicated(["Satellite_ID", "Timestamp"]).sum())
    numeric = frame[required[3:]].apply(pd.to_numeric, errors="coerce")
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype=np.float64))).sum())

    cadence = pd.Timedelta(minutes=contract["time"]["cadence_minutes"])
    diffs = frame.sort_values(["Satellite_ID", "Timestamp"]).groupby("Satellite_ID")["Timestamp"].diff()
    non_cadence_intervals = int(((diffs.notna()) & (diffs != cadence)).sum())

    # Converted SP3 missing clock values are approximately one second. Use a
    # tight neighborhood while accepting the common rounded representation.
    modelled_clock = frame["Modelled_Clock"].to_numpy(dtype=np.float64)
    sentinel_mask = np.isclose(modelled_clock, 1.0, rtol=0.0, atol=2e-9)
    sentinel_count = int(sentinel_mask.sum())

    xyz = frame[["Error_X", "Error_Y", "Error_Z"]].to_numpy(dtype=np.float64)
    derived_3d = np.linalg.norm(xyz, axis=1)
    reported_3d = frame["3D_Orbit_Error"].to_numpy(dtype=np.float64)
    norm_identity_max_abs_error_m = float(np.nanmax(np.abs(derived_3d - reported_3d)))
    kilometre_mask = derived_3d >= 1_000.0
    kilometre_fraction = float(kilometre_mask.mean())
    timestamp_event_fraction = (
        pd.DataFrame({"Timestamp": frame["Timestamp"], "event": kilometre_mask})
        .groupby("Timestamp")["event"]
        .mean()
    )
    synchronous_km_epochs = int((timestamp_event_fraction >= 0.95).sum())

    lag5_correlations = []
    for _, group in frame.assign(_event=kilometre_mask).sort_values("Timestamp").groupby("Satellite_ID"):
        event = group["_event"].to_numpy(dtype=np.float64)
        if len(event) > 5 and event[:-5].std() > 0 and event[5:].std() > 0:
            lag5_correlations.append(float(np.corrcoef(event[:-5], event[5:])[0, 1]))

    critical_failures = []
    if invalid_timestamps:
        critical_failures.append(f"{invalid_timestamps} timestamps cannot be parsed")
    if duplicate_keys:
        critical_failures.append(f"{duplicate_keys} duplicate satellite/timestamp keys")
    if non_finite:
        critical_failures.append(f"{non_finite} non-finite required numeric values")
    if sentinel_count:
        critical_failures.append(
            f"{sentinel_count} SP3 missing-clock sentinels were converted to ~1 second"
        )
    if kilometre_fraction > 0.01 and synchronous_km_epochs:
        critical_failures.append(
            "synchronous kilometre-scale orbit residuals indicate an upstream epoch/alignment defect"
        )

    return {
        "data_path": str(Path(data_path).resolve()),
        "sha256": sha256_file(data_path),
        "rows": int(len(frame)),
        "satellites": int(frame["Satellite_ID"].nunique()),
        "start": frame["Timestamp"].min(),
        "end": frame["Timestamp"].max(),
        "invalid_timestamps": invalid_timestamps,
        "duplicate_satellite_timestamps": duplicate_keys,
        "non_finite_required_values": non_finite,
        "non_cadence_intervals": non_cadence_intervals,
        "sp3_missing_clock_sentinel_rows": sentinel_count,
        "norm_identity_max_abs_error_m": norm_identity_max_abs_error_m,
        "orbit_error_at_least_1km_fraction": kilometre_fraction,
        "synchronous_95pct_1km_epochs": synchronous_km_epochs,
        "mean_event_indicator_lag5_correlation": (
            float(np.mean(lag5_correlations)) if lag5_correlations else None
        ),
        "critical_failures": critical_failures,
        "passed": not critical_failures,
        "policy": {
            "sentinel_rows_must_be_masked": True,
            "target_magnitude_rows_must_not_be_deleted": True,
            "orbit_alignment_must_be_rebuilt_before_scientific_promotion": bool(
                kilometre_fraction > 0.01 and synchronous_km_epochs
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a GNSS CSV against its data contract")
    parser.add_argument("--data", default="FINAL_Data.csv")
    parser.add_argument("--contract", default="configs/data_contract.json")
    parser.add_argument("--report", default="data_quality_report.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on a critical failure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_csv(args.data, args.contract)
    write_json(args.report, report)
    print(json.dumps(report, indent=2, default=str))
    if args.strict and not report["passed"]:
        sys.exit(2)


if __name__ == "__main__":
    main()


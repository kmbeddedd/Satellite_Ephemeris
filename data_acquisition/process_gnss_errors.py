"""Compute GNSS orbit and clock error dataset from Broadcast Ephemeris and Precise SP3 products.

Complies strictly with configs/data_contract.json.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SP3_MISSING_CLOCK_SENTINEL = 999999.999999  # microseconds


def parse_sp3_positions(sp3_path: Path) -> pd.DataFrame:
    """Parse precise satellite positions and clock offsets from an SP3 file.

    Returns DataFrame with columns:
        [Timestamp, Satellite_ID, Modelled_X, Modelled_Y, Modelled_Z, Modelled_Clock]
        (Units: metres, seconds)
    """
    records = []
    current_dt: datetime | None = None

    with open(sp3_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("*"):
                # Epoch header: *  YYYY MM DD HH MM SS.SSSSSSSS
                parts = line[1:].split()
                if len(parts) >= 6:
                    year, month, day, hour, minute = map(int, parts[:5])
                    sec = float(parts[5])
                    current_dt = datetime(
                        year, month, day, hour, minute, int(sec),
                        tzinfo=timezone.utc
                    )
            elif line.startswith("P") and current_dt is not None:
                # Position line: P [SatID] [X_km] [Y_km] [Z_km] [Clock_us]
                # Format: P G01  14234.567890 -12345.678901  21345.678901   123.456789
                sat_id = line[1:4].strip()
                tokens = line[4:].split()
                if len(tokens) >= 4:
                    try:
                        x_km = float(tokens[0])
                        y_km = float(tokens[1])
                        z_km = float(tokens[2])
                        clk_us = float(tokens[3])

                        # SP3 coordinates are in kilometers, convert to meters
                        x_m = x_km * 1000.0
                        y_m = y_km * 1000.0
                        z_m = z_km * 1000.0

                        # SP3 clock is in microseconds; missing sentinel is 999999.999999
                        if math.isclose(clk_us, SP3_MISSING_CLOCK_SENTINEL, abs_tol=1e-3):
                            clk_s = np.nan
                        else:
                            clk_s = clk_us * 1e-6  # Convert to seconds

                        records.append({
                            "Timestamp": current_dt.strftime("%Y-%m-%d %H:%M:%S"),
                            "Satellite_ID": sat_id,
                            "Modelled_X": x_m,
                            "Modelled_Y": y_m,
                            "Modelled_Z": z_m,
                            "Modelled_Clock": clk_s,
                        })
                    except ValueError:
                        continue

    return pd.DataFrame(records)


def assemble_contract_dataset(
    sp3_df: pd.DataFrame,
    broadcast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join broadcast and precise positions, compute errors, and enforce contract schema."""
    merged = pd.merge(
        broadcast_df,
        sp3_df,
        on=["Timestamp", "Satellite_ID"],
        how="inner"
    )

    # Compute errors (Broadcast - Modelled/Precise)
    merged["Error_X"] = merged["Broadcast_X"] - merged["Modelled_X"]
    merged["Error_Y"] = merged["Broadcast_Y"] - merged["Modelled_Y"]
    merged["Error_Z"] = merged["Broadcast_Z"] - merged["Modelled_Z"]

    # 3D Orbit Error is exact Euclidean norm of (Error_X, Error_Y, Error_Z)
    merged["3D_Orbit_Error"] = np.linalg.norm(
        merged[["Error_X", "Error_Y", "Error_Z"]].to_numpy(dtype=np.float64),
        axis=1
    )

    # Clock error
    merged["Error_Clock"] = merged["Broadcast_Clock"] - merged["Modelled_Clock"]

    # Add Constellation column
    merged["Constellation"] = merged["Satellite_ID"].str[0]

    # Required contract column ordering
    required_cols = [
        "Timestamp",
        "Satellite_ID",
        "Constellation",
        "Broadcast_X",
        "Broadcast_Y",
        "Broadcast_Z",
        "Broadcast_Clock",
        "Modelled_X",
        "Modelled_Y",
        "Modelled_Z",
        "Modelled_Clock",
        "Error_X",
        "Error_Y",
        "Error_Z",
        "3D_Orbit_Error",
        "Error_Clock",
    ]

    return merged[required_cols].sort_values(["Satellite_ID", "Timestamp"]).reset_index(drop=True)

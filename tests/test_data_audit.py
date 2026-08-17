from pathlib import Path

import pandas as pd

from audit_data import audit_csv


def test_audit_detects_converted_sp3_clock_sentinel(tmp_path: Path):
    rows = []
    for step in range(3):
        rows.append({
            "Timestamp": f"2026-01-01 00:{step * 15:02d}:00",
            "Satellite_ID": "G01",
            "Constellation": "G",
            "Broadcast_X": 1.0,
            "Broadcast_Y": 2.0,
            "Broadcast_Z": 3.0,
            "Broadcast_Clock": 0.0,
            "Modelled_X": 1.0,
            "Modelled_Y": 2.0,
            "Modelled_Z": 3.0,
            "Modelled_Clock": 1.0 if step == 1 else 0.0,
            "Error_X": 0.0,
            "Error_Y": 0.0,
            "Error_Z": 0.0,
            "3D_Orbit_Error": 0.0,
            "Error_Clock": -1.0 if step == 1 else 0.0,
        })
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = audit_csv(str(csv_path))
    assert report["sp3_missing_clock_sentinel_rows"] == 1
    assert not report["passed"]


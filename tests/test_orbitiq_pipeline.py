import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_acquisition.build_orbitiq_benchmark import generate_orbitiq_benchmark
from audit_data import audit_csv
from train_orbitiq_pipeline import run_pipeline


def test_orbitiq_benchmark_generation_and_audit(tmp_path):
    benchmark_path = tmp_path / "test_orbitiq_benchmark.csv"
    df = generate_orbitiq_benchmark(
        data_dir=Path("data/orbitiq"),
        output_path=benchmark_path,
    )
    assert len(df) == 2304
    assert set(df["Satellite_ID"].unique()) == {"GEO01", "MEO01", "MEO02"}
    assert "Error_X" in df.columns
    assert "Error_Y" in df.columns
    assert "Error_Z" in df.columns
    assert "Error_Clock" in df.columns

    report = audit_csv(str(benchmark_path))
    assert report["passed"] is True
    assert report["rows"] == 2304
    assert report["satellites"] == 3
    assert report["sp3_missing_clock_sentinel_rows"] == 0
    assert report["non_cadence_intervals"] == 0


def test_orbitiq_end_to_end_pipeline_fast(tmp_path):
    output_dir = tmp_path / "pipeline_out"
    benchmark_path = tmp_path / "test_orbitiq_benchmark.csv"
    
    report = run_pipeline(
        raw_data_dir=Path("data/orbitiq"),
        benchmark_path=benchmark_path,
        output_dir=output_dir,
        epochs=1,
        batch_size=32,
        lr=1e-3,
        seed=42,
        skip_plots=True,
    )

    assert "bilstm_overall_3d_mae" in report
    assert "transformer_overall_3d_mae" in report
    assert "conformal_calibration" in report
    assert report["conformal_calibration"]["coverage_90_pct"] > 0
    assert report["conformal_calibration"]["coverage_95_pct"] > 0
    assert (output_dir / "pipeline_metrics.json").exists()
    assert (output_dir / "ORBITIQ_PIPELINE_REPORT.md").exists()

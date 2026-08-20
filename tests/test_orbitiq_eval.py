from pathlib import Path
import sys
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate_orbitiq import preprocess_data, evaluate_orbitiq_dataset, FEATURE_COLUMNS


def test_orbitiq_data_files_exist():
    data_dir = Path("data/orbitiq")
    assert (data_dir / "DATA_GEO_Train.csv").exists()
    assert (data_dir / "DATA_MEO_Train.csv").exists()
    assert (data_dir / "DATA_MEO_Train2.csv").exists()


def test_orbitiq_pretrained_models_exist():
    models_dir = Path("models/orbitiq_pretrained")
    assert (models_dir / "scalers.pkl").exists()
    for orbit in ["GEO", "MEO"]:
        for col in FEATURE_COLUMNS:
            assert (models_dir / f"{orbit}_{col}_model.h5").exists()


def test_orbitiq_preprocessing():
    df = preprocess_data("data/orbitiq/DATA_GEO_Train.csv")
    assert len(df) > 50
    assert "utc_time" in df.columns
    assert "total_position_error" in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
        assert f"{col}_rolling_mean" in df.columns
        assert f"{col}_rolling_std" in df.columns


def test_orbitiq_evaluation_pipeline():
    res = evaluate_orbitiq_dataset(
        data_path="data/orbitiq/DATA_GEO_Train.csv",
        orbit_type="GEO",
        models_dir="models/orbitiq_pretrained",
        seq_length=7,
        test_size=0.2
    )
    assert res["dataset"] == "DATA_GEO_Train.csv"
    assert "lstm_pretrained" in res
    assert "random_forest" in res
    assert "3d_position_error_mean_m" in res["lstm_pretrained"]
    assert res["lstm_pretrained"]["3d_position_error_mean_m"] > 0.0

    for col in FEATURE_COLUMNS:
        assert col in res["lstm_pretrained"]
        m = res["lstm_pretrained"][col]
        assert "mae" in m
        assert "rmse" in m
        assert "shapiro_w" in m
        assert "is_normal" in m

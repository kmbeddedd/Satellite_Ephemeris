"""Run zero, persistence, seasonal, and drift GNSS forecast baselines.

Example:

    python evaluate_baselines.py --data FINAL_Data.csv --horizon 96 \
        --output baseline_metrics.json

The final ``horizon`` rows for each eligible satellite are labels; history is
taken strictly from preceding rows. Promotion failures are reported in JSON and
stdout. They only produce a non-zero exit status when ``--strict-promotion`` is
explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.baselines import BASELINE_NAMES, evaluate_baselines
from src.config import HORIZON_MAP, TARGET_COLS_4
from src.evaluate import compare_candidate_to_baseline, evaluate_forecasts


EXPECTED_INTERVAL = pd.Timedelta(minutes=15)
SP3_CLOCK_SENTINEL_SECONDS = 0.999999999999
SP3_CLOCK_SENTINEL_ATOL = 1e-9


DEFAULT_PROMOTION_RULES = {
    "all_forecast_points.orbit_3d_vector_error.mae": {
        "direction": "lower",
        "min_relative_improvement": 0.0,
    },
    "all_forecast_points.clock_error.seconds.mae": {
        "direction": "lower",
        "min_relative_improvement": 0.0,
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate executable GNSS residual forecasting baselines"
    )
    parser.add_argument("--data", required=True, help="Input CSV ordered by satellite and time")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--horizon", type=int, default=96, help="Final rows per satellite to score")
    parser.add_argument(
        "--lookback",
        type=int,
        help="History rows immediately before the labels (default: enough for requested baselines)",
    )
    parser.add_argument("--season-length", type=int, default=96)
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(BASELINE_NAMES),
    )
    parser.add_argument(
        "--targets", nargs="+", choices=TARGET_COLS_4, default=list(TARGET_COLS_4)
    )
    parser.add_argument("--satellite-column", default="Satellite_ID")
    parser.add_argument("--constellation-column", default="Constellation")
    parser.add_argument("--time-column", default="Timestamp")
    parser.add_argument(
        "--candidate-predictions",
        help="Optional .npy/.npz candidate predictions aligned to the reported satellite order",
    )
    parser.add_argument(
        "--promotion-baseline",
        choices=BASELINE_NAMES,
        default="zero",
        help="Baseline used for the optional promotion comparison",
    )
    parser.add_argument(
        "--promotion-config",
        help="Optional JSON mapping of metric paths to promotion rules",
    )
    parser.add_argument(
        "--strict-promotion",
        action="store_true",
        help="Return exit code 2 when candidate promotion gates fail",
    )
    return parser.parse_args(argv)


def _load_backtest_arrays(
    data_path: str,
    target_cols: Sequence[str],
    satellite_column: str,
    constellation_column: str,
    time_column: str,
    horizon: int,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], int]:
    if horizon < 1 or lookback < 1:
        raise ValueError("horizon and lookback must be positive")
    frame = pd.read_csv(data_path)
    required = {satellite_column, time_column, *target_cols}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")
    frame[time_column] = pd.to_datetime(frame[time_column], errors="raise")
    frame = frame.sort_values([satellite_column, time_column], kind="stable")

    # Apply the same target validity contract as the training data path without
    # importing its torch-dependent dataset builders into this lightweight CLI.
    for target in target_cols:
        numeric = pd.to_numeric(frame[target], errors="coerce")
        valid = np.isfinite(numeric.to_numpy(dtype=np.float64, na_value=np.nan))
        supplied_validity = f"{target}_valid"
        if supplied_validity in frame.columns:
            valid &= frame[supplied_validity].fillna(False).astype(bool).to_numpy()
        frame[target] = numeric.astype(float)
        frame[f"__{target}_valid"] = valid

    if "Error_Clock" in target_cols:
        sentinel = np.zeros(len(frame), dtype=bool)
        source_found = False
        for source in ("Modelled_Clock", "Precise_Clock", "SP3_Clock"):
            if source in frame.columns:
                values = pd.to_numeric(frame[source], errors="coerce").to_numpy(float)
                sentinel |= np.isclose(
                    values,
                    SP3_CLOCK_SENTINEL_SECONDS,
                    rtol=0.0,
                    atol=SP3_CLOCK_SENTINEL_ATOL,
                )
                source_found = True
        if not source_found and {"Broadcast_Clock", "Error_Clock"}.issubset(frame.columns):
            broadcast = pd.to_numeric(frame["Broadcast_Clock"], errors="coerce").to_numpy(float)
            clock_error = pd.to_numeric(frame["Error_Clock"], errors="coerce").to_numpy(float)
            sentinel |= np.isclose(
                broadcast - clock_error,
                SP3_CLOCK_SENTINEL_SECONDS,
                rtol=0.0,
                atol=SP3_CLOCK_SENTINEL_ATOL,
            )
            source_found = True
        if not source_found:
            clock_error = pd.to_numeric(frame["Error_Clock"], errors="coerce").to_numpy(float)
            sentinel |= np.isclose(
                np.abs(clock_error),
                SP3_CLOCK_SENTINEL_SECONDS,
                rtol=0.0,
                atol=SP3_CLOCK_SENTINEL_ATOL,
            )
        frame.loc[sentinel, "__Error_Clock_valid"] = False

    for target in target_cols:
        frame.loc[~frame[f"__{target}_valid"], target] = np.nan

    histories = []
    actuals = []
    actual_masks = []
    satellite_ids: list[str] = []
    constellations: list[str] = []
    skipped_noncontiguous = 0
    for satellite, satellite_frame in frame.groupby(satellite_column, sort=True):
        if len(satellite_frame) < lookback + horizon:
            continue
        selected = satellite_frame.iloc[-(lookback + horizon) :]
        timestamps = selected[time_column].to_numpy(dtype="datetime64[ns]")
        if not np.all(np.diff(timestamps) == EXPECTED_INTERVAL.to_timedelta64()):
            skipped_noncontiguous += 1
            continue
        values = selected[list(target_cols)].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
        validity = selected[[f"__{target}_valid" for target in target_cols]].to_numpy(
            dtype=bool
        )
        histories.append(values[:-horizon])
        actuals.append(values[-horizon:])
        actual_masks.append(validity[-horizon:])
        satellite_ids.append(str(satellite))
        if constellation_column in satellite_frame.columns:
            constellation = str(satellite_frame[constellation_column].iloc[-1])
        else:
            constellation = str(satellite)[0].upper()
        constellations.append(constellation)

    if not satellite_ids:
        raise ValueError(
            f"no satellite has the required {lookback} history + {horizon} label rows"
        )
    return (
        np.stack(histories),
        np.stack(actuals),
        np.stack(actual_masks),
        satellite_ids,
        constellations,
        skipped_noncontiguous,
    )


def _load_predictions(path: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            for key in ("predictions", "prediction", "pred", "mu"):
                if key in loaded.files:
                    return np.asarray(loaded[key], dtype=np.float64)
            if len(loaded.files) == 1:
                return np.asarray(loaded[loaded.files[0]], dtype=np.float64)
            raise ValueError(
                "candidate NPZ must contain predictions/prediction/pred/mu or exactly one array"
            )
        finally:
            loaded.close()
    return np.asarray(loaded, dtype=np.float64)


def _load_rules(path: Optional[str], target_cols: Sequence[str]) -> Mapping[str, Any]:
    if path:
        with open(path, "r", encoding="utf-8") as file_handle:
            rules = json.load(file_handle)
        if not isinstance(rules, Mapping):
            raise ValueError("promotion config must be a JSON object")
        return rules
    rules = dict(DEFAULT_PROMOTION_RULES)
    if not all(target in target_cols for target in ("Error_X", "Error_Y", "Error_Z")):
        rules.pop("all_forecast_points.orbit_3d_vector_error.mae")
    if "Error_Clock" not in target_cols:
        rules.pop("all_forecast_points.clock_error.seconds.mae")
    if not rules:
        raise ValueError("provide --promotion-config when default orbit/clock targets are absent")
    return rules


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _headline(report: Mapping[str, Any]) -> str:
    block = report["all_forecast_points"]
    values = []
    if "orbit_3d_vector_error" in block:
        values.append(f"3D MAE={block['orbit_3d_vector_error']['mae']:.6g} m")
    if "clock_error" in block:
        values.append(f"clock MAE={block['clock_error']['nanoseconds']['mae']:.6g} ns")
    return ", ".join(values)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    minimum_history = args.season_length if "seasonal" in args.baselines else 2
    lookback = args.lookback if args.lookback is not None else minimum_history
    if "seasonal" in args.baselines and lookback < args.season_length:
        raise ValueError("lookback must be at least season_length for the seasonal baseline")

    history, actual, actual_mask, satellite_ids, constellations, skipped_noncontiguous = (
        _load_backtest_arrays(
        args.data,
        args.targets,
        args.satellite_column,
        args.constellation_column,
        args.time_column,
        args.horizon,
        lookback,
        )
    )
    horizons = {label: step for label, step in HORIZON_MAP.items() if step <= args.horizon}
    if args.horizon not in horizons.values():
        horizons[f"lead {args.horizon}"] = args.horizon
    baseline_reports = evaluate_baselines(
        history,
        actual,
        args.targets,
        baselines=args.baselines,
        season_length=args.season_length,
        horizons=horizons,
        satellite_ids=satellite_ids,
        constellations=constellations,
        valid_mask=actual_mask,
    )
    report: Dict[str, Any] = {
        "metadata": {
            "data": str(Path(args.data).resolve()),
            "satellites": satellite_ids,
            "target_cols": list(args.targets),
            "lookback_steps": lookback,
            "forecast_steps": args.horizon,
            "label_policy": "final horizon rows per satellite; history strictly precedes labels",
            "cadence_contract": "every scored history+label block has exact 15-minute steps",
            "skipped_noncontiguous_satellites": skipped_noncontiguous,
            "valid_label_count_by_target": {
                target: int(actual_mask[..., index].sum())
                for index, target in enumerate(args.targets)
            },
        },
        "baselines": baseline_reports,
    }

    for name, baseline_report in baseline_reports.items():
        print(f"{name:>11}: {_headline(baseline_report)}")

    promotion_failed = False
    if args.candidate_predictions:
        candidate_predictions = _load_predictions(args.candidate_predictions)
        candidate_report = evaluate_forecasts(
            actual,
            candidate_predictions,
            args.targets,
            horizons=horizons,
            satellite_ids=satellite_ids,
            constellations=constellations,
            valid_mask=actual_mask,
        )
        if args.promotion_baseline not in baseline_reports:
            raise ValueError(
                f"promotion baseline {args.promotion_baseline!r} was not among --baselines"
            )
        rules = _load_rules(args.promotion_config, args.targets)
        promotion = compare_candidate_to_baseline(
            candidate_report,
            baseline_reports[args.promotion_baseline],
            rules,
        )
        promotion_failed = not promotion["passed"]
        report["candidate"] = candidate_report
        report["promotion"] = {
            "baseline": args.promotion_baseline,
            "rules": rules,
            **promotion,
        }
        status = "PASS" if promotion["passed"] else "FAIL (reported; artifacts remain usable)"
        print(f"  promotion: {status}")
        for comparison in promotion["comparisons"]:
            print(
                f"    {comparison['metric']}: "
                f"{'PASS' if comparison['passed'] else 'FAIL'}"
            )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(_json_safe(report), file_handle, indent=2, allow_nan=False)
        print(f"report saved: {output_path}")

    if promotion_failed and args.strict_promotion:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

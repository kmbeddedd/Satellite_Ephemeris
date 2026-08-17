"""Deterministic forecasting baselines for GNSS residual targets.

All functions accept history shaped ``(time, target)`` or
``(series, time, target)`` and preserve that batching convention in the
returned forecast. A one-dimensional single-target history is also accepted.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _normalise_history(history: np.ndarray) -> Tuple[np.ndarray, int]:
    values = np.asarray(history, dtype=np.float64)
    original_ndim = values.ndim
    if original_ndim == 1:
        values = values[None, :, None]
    elif original_ndim == 2:
        values = values[None, :, :]
    elif original_ndim != 3:
        raise ValueError(
            "history must have shape (time,), (time, target), or "
            f"(series, time, target); got {values.shape}"
        )
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError(f"history cannot have an empty dimension; got {values.shape}")
    return values, original_ndim


def _restore_forecast_shape(forecast: np.ndarray, original_ndim: int) -> np.ndarray:
    if original_ndim == 1:
        return forecast[0, :, 0]
    if original_ndim == 2:
        return forecast[0]
    return forecast


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    return int(horizon)


def _last_finite(values: np.ndarray) -> np.ndarray:
    """Return the last finite value for each series/target, or NaN if absent."""

    result = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float64)
    for series_index in range(values.shape[0]):
        for target_index in range(values.shape[2]):
            valid_indices = np.flatnonzero(np.isfinite(values[series_index, :, target_index]))
            if valid_indices.size:
                result[series_index, target_index] = values[
                    series_index, valid_indices[-1], target_index
                ]
    return result


def zero_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast zero correction at every lead and target."""

    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    forecast = np.zeros((values.shape[0], steps, values.shape[2]), dtype=np.float64)
    return _restore_forecast_shape(forecast, original_ndim)


def persistence_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat the last finite observation for every target and future lead."""

    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    forecast = np.repeat(_last_finite(values)[:, None, :], steps, axis=1)
    return _restore_forecast_shape(forecast, original_ndim)


def seasonal_forecast(
    history: np.ndarray,
    horizon: int,
    season_length: int = 96,
) -> np.ndarray:
    """Repeat the final complete season, cycling if horizon exceeds it."""

    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    if (
        isinstance(season_length, bool)
        or not isinstance(season_length, (int, np.integer))
        or int(season_length) < 1
    ):
        raise ValueError("season_length must be a positive integer")
    season_length = int(season_length)
    if values.shape[1] < season_length:
        raise ValueError(
            f"seasonal baseline needs at least {season_length} history steps; "
            f"got {values.shape[1]}"
        )
    final_season = values[:, -season_length:, :]
    indices = np.arange(steps) % season_length
    forecast = final_season[:, indices, :]
    # A missing historical SP3 clock must not become a synthetic one-second
    # seasonal forecast. Fall back to the last finite observation per target.
    fallback = np.repeat(_last_finite(values)[:, None, :], steps, axis=1)
    forecast = np.where(np.isfinite(forecast), forecast, fallback)
    return _restore_forecast_shape(forecast, original_ndim)


def drift_forecast(history: np.ndarray, horizon: int) -> np.ndarray:
    """Extrapolate the line from the first to last history observation."""

    values, original_ndim = _normalise_history(history)
    steps = _validate_horizon(horizon)
    if values.shape[1] < 2:
        raise ValueError("drift baseline needs at least two history steps")
    slope = np.full((values.shape[0], values.shape[2]), np.nan, dtype=np.float64)
    anchor = _last_finite(values)
    for series_index in range(values.shape[0]):
        for target_index in range(values.shape[2]):
            valid_indices = np.flatnonzero(np.isfinite(values[series_index, :, target_index]))
            if valid_indices.size == 1:
                slope[series_index, target_index] = 0.0
            elif valid_indices.size > 1:
                first_index, last_index = valid_indices[0], valid_indices[-1]
                slope[series_index, target_index] = (
                    values[series_index, last_index, target_index]
                    - values[series_index, first_index, target_index]
                ) / float(last_index - first_index)
    lead = np.arange(1, steps + 1, dtype=np.float64)[None, :, None]
    forecast = anchor[:, None, :] + lead * slope[:, None, :]
    return _restore_forecast_shape(forecast, original_ndim)


# Verb-first aliases are convenient in model comparison scripts.
forecast_zero = zero_forecast
forecast_persistence = persistence_forecast
forecast_seasonal = seasonal_forecast
forecast_drift = drift_forecast


BASELINE_NAMES = ("zero", "persistence", "seasonal", "drift")


def generate_baseline_forecasts(
    history: np.ndarray,
    horizon: int,
    baselines: Sequence[str] = BASELINE_NAMES,
    season_length: int = 96,
) -> Dict[str, np.ndarray]:
    """Generate any requested set of named baseline forecasts."""

    generators = {
        "zero": lambda: zero_forecast(history, horizon),
        "persistence": lambda: persistence_forecast(history, horizon),
        "seasonal": lambda: seasonal_forecast(history, horizon, season_length),
        "drift": lambda: drift_forecast(history, horizon),
    }
    requested = [str(name).lower() for name in baselines]
    unknown = sorted(set(requested) - set(generators))
    if unknown:
        raise ValueError(
            f"unknown baselines {unknown}; supported baselines are {list(BASELINE_NAMES)}"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("baseline names must be unique")
    return {name: generators[name]() for name in requested}


def evaluate_baselines(
    history: np.ndarray,
    actual: np.ndarray,
    target_cols: Sequence[str],
    baselines: Sequence[str] = BASELINE_NAMES,
    season_length: int = 96,
    horizons: Optional[Mapping[str, int]] = None,
    satellite_ids: Optional[Sequence[Any]] = None,
    constellations: Optional[Sequence[Any]] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Generate and score baselines through the canonical evaluation API."""

    # Imported lazily to keep baseline generation independently reusable.
    from src.evaluate import evaluate_forecasts

    actual_values = np.asarray(actual, dtype=np.float64)
    if actual_values.ndim not in (1, 2, 3):
        raise ValueError("actual has an unsupported shape")
    forecast_horizon = actual_values.shape[-2] if actual_values.ndim >= 2 else actual_values.shape[0]
    forecasts = generate_baseline_forecasts(
        history,
        forecast_horizon,
        baselines=baselines,
        season_length=season_length,
    )
    return {
        name: evaluate_forecasts(
            actual_values,
            forecast,
            target_cols,
            horizons=horizons,
            satellite_ids=satellite_ids,
            constellations=constellations,
            valid_mask=valid_mask,
        )
        for name, forecast in forecasts.items()
    }

"""Unit-aware deterministic and probabilistic forecast evaluation.

The public :func:`evaluate_forecasts` API is the canonical evaluator. It keeps
orbit (metres) and clock (seconds) scores separate, derives three-dimensional
orbit error from the XYZ residual vector, and reports both exact-lead and
cumulative-window horizon semantics explicitly.

The older ``compute_*`` functions at the bottom of this module are retained as
compatibility adapters for the original training and plotting scripts.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t_distribution

from src.config import HORIZON_MAP, TARGET_COLS_5


SPEED_OF_LIGHT_M_S = 299_792_458.0
XYZ_TARGETS = ("Error_X", "Error_Y", "Error_Z")
DERIVED_3D_TARGETS = frozenset({"3D_Orbit_Error", "Orbit_3D_Error"})


class PromotionGateError(AssertionError):
    """Raised when a candidate fails a configured baseline promotion gate."""


@dataclass(frozen=True)
class PromotionRule:
    """One fail-closed candidate-versus-baseline metric comparison.

    ``metric`` is a dot-separated path into an evaluation report. Use
    ``direction='lower'`` for error/loss metrics and ``direction='higher'``
    for skill or coverage metrics.
    """

    metric: str
    direction: str = "lower"
    min_absolute_improvement: float = 0.0
    min_relative_improvement: float = 0.0
    allow_equal: bool = False


def _as_forecast_tensor(values: np.ndarray, name: str) -> Tuple[np.ndarray, bool]:
    """Return ``(N, horizon, targets)`` float data and whether N was implicit."""

    array = np.asarray(values, dtype=np.float64)
    implicit_series = False
    if array.ndim == 1:
        array = array[None, :, None]
        implicit_series = True
    elif array.ndim == 2:
        array = array[None, :, :]
        implicit_series = True
    if array.ndim != 3:
        raise ValueError(
            f"{name} must have shape (horizon, targets) or "
            f"(series, horizon, targets); got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} cannot have an empty dimension; got {array.shape}")
    return array, implicit_series


def _validate_forecasts(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_cols: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    actual_tensor, _ = _as_forecast_tensor(actual, "actual")
    predicted_tensor, _ = _as_forecast_tensor(predicted, "predicted")
    if actual_tensor.shape != predicted_tensor.shape:
        raise ValueError(
            "actual and predicted must have identical shapes; "
            f"got {actual_tensor.shape} and {predicted_tensor.shape}"
        )
    columns = list(target_cols)
    if len(columns) != actual_tensor.shape[-1]:
        raise ValueError(
            f"target_cols has {len(columns)} names but arrays have "
            f"{actual_tensor.shape[-1]} targets"
        )
    if len(set(columns)) != len(columns):
        raise ValueError("target_cols must contain unique names")
    return actual_tensor, predicted_tensor, columns


def _mask_actual(actual: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    """Represent explicitly invalid labels as NaN for all downstream metrics."""

    if valid_mask is None:
        return actual
    mask_values = np.asarray(valid_mask)
    if mask_values.ndim == 1:
        mask_values = mask_values[None, :, None]
    elif mask_values.ndim == 2:
        # Prefer a one-series (H, T) interpretation when shapes match; a
        # (N, H) mask may still broadcast by supplying its final singleton.
        if tuple(mask_values.shape) == tuple(actual.shape[1:]):
            mask_values = mask_values[None, :, :]
        elif tuple(mask_values.shape) == tuple(actual.shape[:2]):
            mask_values = mask_values[:, :, None]
    try:
        mask_values = np.broadcast_to(mask_values, actual.shape)
    except ValueError as error:
        raise ValueError(
            f"valid_mask must be broadcastable to forecast shape {actual.shape}; "
            f"got {np.asarray(valid_mask).shape}"
        ) from error
    if mask_values.dtype == np.bool_:
        valid = mask_values
    else:
        valid = np.isfinite(mask_values) & (mask_values != 0)
    return np.where(valid, actual, np.nan)


def _unit_for_target(target: str) -> str:
    lowered = target.lower()
    if "clock" in lowered:
        return "s"
    if target in XYZ_TARGETS or "orbit" in lowered:
        return "m"
    return "native"


def _empty_error_summary() -> Dict[str, Any]:
    return {
        "count": 0,
        "mae": None,
        "rmse": None,
        "median_ae": None,
        "p90_ae": None,
        "p95_ae": None,
        "p99_ae": None,
    }


def compute_error_metrics(residuals: np.ndarray) -> Dict[str, Any]:
    """Summarize signed residuals, omitting non-finite values pairwise."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return _empty_error_summary()
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median_ae": float(np.median(absolute)),
        "p90_ae": float(np.percentile(absolute, 90)),
        "p95_ae": float(np.percentile(absolute, 95)),
        "p99_ae": float(np.percentile(absolute, 99)),
    }


def _scaled_error_summary(residuals: np.ndarray, scale: float) -> Dict[str, Any]:
    return compute_error_metrics(np.asarray(residuals, dtype=np.float64) * scale)


def _paired_residual(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    paired = np.isfinite(actual) & np.isfinite(predicted)
    return np.where(paired, residual, np.nan)


def _xyz_indices(target_cols: Sequence[str]) -> Optional[Tuple[int, int, int]]:
    if all(target in target_cols for target in XYZ_TARGETS):
        return tuple(target_cols.index(target) for target in XYZ_TARGETS)  # type: ignore[return-value]
    return None


def _clock_index(target_cols: Sequence[str], clock_col: str) -> Optional[int]:
    return target_cols.index(clock_col) if clock_col in target_cols else None


def _deterministic_score_block(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_cols: Sequence[str],
    clock_col: str,
) -> Dict[str, Any]:
    """Score selected forecast points without mixing measurement units."""

    per_target: Dict[str, Dict[str, Any]] = {}
    for index, target in enumerate(target_cols):
        # A supplied scalar 3D target is deliberately not scored. The physical
        # vector error below is always derived from XYZ forecast residuals.
        if target in DERIVED_3D_TARGETS:
            continue
        summary = compute_error_metrics(_paired_residual(actual[..., index], predicted[..., index]))
        per_target[target] = {"unit": _unit_for_target(target), **summary}

    block: Dict[str, Any] = {"per_target": per_target}

    xyz_indices = _xyz_indices(target_cols)
    if xyz_indices is not None:
        actual_xyz = actual[..., list(xyz_indices)]
        predicted_xyz = predicted[..., list(xyz_indices)]
        vector_residual = predicted_xyz - actual_xyz
        valid = np.all(np.isfinite(actual_xyz) & np.isfinite(predicted_xyz), axis=-1)
        vector_error = np.where(valid, np.linalg.norm(vector_residual, axis=-1), np.nan)
        block["orbit_3d_vector_error"] = {
            "unit": "m",
            "derived_from": list(XYZ_TARGETS),
            **compute_error_metrics(vector_error),
        }

    clock_idx = _clock_index(target_cols, clock_col)
    if clock_idx is not None:
        clock_residual = _paired_residual(actual[..., clock_idx], predicted[..., clock_idx])
        block["clock_error"] = {
            "seconds": {"unit": "s", **compute_error_metrics(clock_residual)},
            "nanoseconds": {
                "unit": "ns",
                **_scaled_error_summary(clock_residual, 1e9),
            },
            "range_equivalent_metres": {
                "unit": "m",
                "conversion": "c * dt",
                "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
                **_scaled_error_summary(clock_residual, SPEED_OF_LIGHT_M_S),
            },
        }

    return block


def _normalise_horizons(horizons: Optional[Mapping[str, int]], horizon: int) -> Dict[str, int]:
    if horizons is None:
        return {f"lead_{horizon}": horizon}
    result: Dict[str, int] = {}
    for label, step in horizons.items():
        if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 1:
            raise ValueError(f"Horizon step for {label!r} must be a positive integer")
        result[str(label)] = int(step)
    return result


def _deterministic_core_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_cols: Sequence[str],
    horizons: Optional[Mapping[str, int]],
    clock_col: str,
) -> Dict[str, Any]:
    available_horizon = actual.shape[1]
    report: Dict[str, Any] = {
        "protocol": {
            "array_shape": "series, forecast_lead, target",
            "available_forecast_steps": int(available_horizon),
            "exact_lead_definition": "score only forecast step h (zero-based index h-1)",
            "cumulative_window_definition": "score forecast steps 1 through h inclusive",
            "three_dimensional_orbit_error": "derived as ||predicted_XYZ - actual_XYZ||_2",
            "unit_policy": "no aggregate combines orbit metres with clock seconds",
        },
        "all_forecast_points": _deterministic_score_block(
            actual, predicted, target_cols, clock_col
        ),
        "horizons": {},
    }

    for label, step in _normalise_horizons(horizons, available_horizon).items():
        if step > available_horizon:
            report["horizons"][label] = {
                "requested_lead_step": step,
                "available": False,
                "reason": f"only {available_horizon} forecast steps are available",
                "exact_lead": None,
                "cumulative_window_1_to_lead": None,
            }
            continue
        report["horizons"][label] = {
            "requested_lead_step": step,
            "available": True,
            "exact_lead": _deterministic_score_block(
                actual[:, step - 1 : step, :],
                predicted[:, step - 1 : step, :],
                target_cols,
                clock_col,
            ),
            "cumulative_window_1_to_lead": _deterministic_score_block(
                actual[:, :step, :], predicted[:, :step, :], target_cols, clock_col
            ),
        }
    return report


def _normalise_slice_labels(
    labels: Optional[Sequence[Any]], n_series: int, name: str
) -> Optional[np.ndarray]:
    if labels is None:
        return None
    values = np.asarray(labels, dtype=object).reshape(-1)
    if values.size != n_series:
        raise ValueError(f"{name} must contain one value per series ({n_series}); got {values.size}")
    return values.astype(str)


def infer_constellation(satellite_id: str) -> str:
    """Infer the conventional one-character constellation code from a PRN."""

    satellite = str(satellite_id).strip()
    return satellite[0].upper() if satellite else "UNKNOWN"


def evaluate_forecasts(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_cols: Sequence[str],
    horizons: Optional[Mapping[str, int]] = HORIZON_MAP,
    satellite_ids: Optional[Sequence[Any]] = None,
    constellations: Optional[Sequence[Any]] = None,
    clock_col: str = "Error_Clock",
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute a unified, unit-aware deterministic forecast report.

    Parameters are arrays shaped ``(series, horizon, targets)``. A two-
    dimensional ``(horizon, targets)`` forecast is accepted as one series.
    If ``satellite_ids`` are supplied, both per-satellite and per-constellation
    slice reports are produced. Constellations are inferred from PRN prefixes
    unless the explicit ``constellations`` sequence is given.
    """

    actual_tensor, predicted_tensor, columns = _validate_forecasts(
        actual, predicted, target_cols
    )
    actual_tensor = _mask_actual(actual_tensor, valid_mask)
    report = _deterministic_core_report(
        actual_tensor, predicted_tensor, columns, horizons, clock_col
    )

    sat_labels = _normalise_slice_labels(satellite_ids, actual_tensor.shape[0], "satellite_ids")
    constellation_labels = _normalise_slice_labels(
        constellations, actual_tensor.shape[0], "constellations"
    )
    if constellation_labels is not None and sat_labels is None:
        raise ValueError("constellations can only be supplied together with satellite_ids")

    if sat_labels is not None:
        if constellation_labels is None:
            constellation_labels = np.asarray(
                [infer_constellation(satellite) for satellite in sat_labels], dtype=object
            )
        slices: Dict[str, Dict[str, Any]] = {"per_satellite": {}, "per_constellation": {}}
        for satellite in sorted(set(sat_labels.tolist())):
            mask = sat_labels == satellite
            slices["per_satellite"][satellite] = _deterministic_core_report(
                actual_tensor[mask], predicted_tensor[mask], columns, horizons, clock_col
            )
        for constellation in sorted(set(constellation_labels.tolist())):
            mask = constellation_labels == constellation
            slices["per_constellation"][str(constellation)] = _deterministic_core_report(
                actual_tensor[mask], predicted_tensor[mask], columns, horizons, clock_col
            )
        report["slices"] = slices

    return report


def _validate_sigma(actual: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma_tensor, _ = _as_forecast_tensor(sigma, "sigma")
    if sigma_tensor.shape != actual.shape:
        raise ValueError(
            f"sigma must match actual/mean shape {actual.shape}; got {sigma_tensor.shape}"
        )
    invalid = np.isfinite(sigma_tensor) & (sigma_tensor <= 0)
    if np.any(invalid):
        raise ValueError("all finite predictive standard deviations must be positive")
    return sigma_tensor


def _gaussian_target_metrics(
    actual: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    coverages: Sequence[float],
    min_sigma: float,
) -> Dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(mean) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return {"count": 0, "gaussian_nll": None, "intervals": {}}
    target = actual[valid]
    location = mean[valid]
    scale = np.maximum(sigma[valid], min_sigma)
    squared_standard_error = np.square((target - location) / scale)
    nll = 0.5 * (math.log(2.0 * math.pi) + 2.0 * np.log(scale) + squared_standard_error)
    intervals: Dict[str, Dict[str, float]] = {}
    normal = NormalDist()
    for coverage in coverages:
        if not 0.0 < float(coverage) < 1.0:
            raise ValueError("coverage levels must lie strictly between zero and one")
        z_score = normal.inv_cdf((1.0 + float(coverage)) / 2.0)
        lower = location - z_score * scale
        upper = location + z_score * scale
        label = f"{100.0 * float(coverage):g}%"
        intervals[label] = {
            "nominal_coverage": float(coverage),
            "empirical_coverage": float(np.mean((target >= lower) & (target <= upper))),
            "mean_width": float(np.mean(upper - lower)),
        }
    return {
        "count": int(target.size),
        "gaussian_nll": float(np.mean(nll)),
        "intervals": intervals,
    }


def _student_t_target_metrics(
    actual: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: np.ndarray,
    coverages: Sequence[float],
    min_scale: float,
) -> Dict[str, Any]:
    valid = (
        np.isfinite(actual)
        & np.isfinite(mean)
        & np.isfinite(scale)
        & (scale > 0)
        & np.isfinite(degrees_of_freedom)
        & (degrees_of_freedom > 0)
    )
    if not np.any(valid):
        return {"count": 0, "student_t_nll": None, "intervals": {}}
    target = actual[valid]
    location = mean[valid]
    target_scale = np.maximum(scale[valid], min_scale)
    target_df = degrees_of_freedom[valid]
    standard_error = (target - location) / target_scale
    log_density = (
        gammaln((target_df + 1.0) / 2.0)
        - gammaln(target_df / 2.0)
        - 0.5 * np.log(target_df * math.pi)
        - np.log(target_scale)
        - ((target_df + 1.0) / 2.0)
        * np.log1p(np.square(standard_error) / target_df)
    )
    intervals: Dict[str, Dict[str, float]] = {}
    for coverage in coverages:
        if not 0.0 < float(coverage) < 1.0:
            raise ValueError("coverage levels must lie strictly between zero and one")
        quantile = student_t_distribution.ppf(
            (1.0 + float(coverage)) / 2.0, target_df
        )
        lower = location - quantile * target_scale
        upper = location + quantile * target_scale
        label = f"{100.0 * float(coverage):g}%"
        intervals[label] = {
            "nominal_coverage": float(coverage),
            "empirical_coverage": float(np.mean((target >= lower) & (target <= upper))),
            "mean_width": float(np.mean(upper - lower)),
        }
    return {
        "count": int(target.size),
        "student_t_nll": float(np.mean(-log_density)),
        "mean_degrees_of_freedom": float(np.mean(target_df)),
        "intervals": intervals,
    }


def _scale_interval_widths(metrics: Mapping[str, Any], scale: float, unit: str) -> Dict[str, Any]:
    intervals = {
        label: {
            **values,
            "mean_width": float(values["mean_width"] * scale),
            "unit": unit,
        }
        for label, values in metrics.get("intervals", {}).items()
    }
    return {"count": metrics.get("count", 0), "unit": unit, "intervals": intervals}


def _gaussian_score_block(
    actual: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    target_cols: Sequence[str],
    clock_col: str,
    coverages: Sequence[float],
    min_sigma: float,
) -> Dict[str, Any]:
    per_target: Dict[str, Any] = {}
    raw_metrics: Dict[str, Dict[str, Any]] = {}
    for index, target in enumerate(target_cols):
        if target in DERIVED_3D_TARGETS:
            continue
        metrics = _gaussian_target_metrics(
            actual[..., index], mean[..., index], sigma[..., index], coverages, min_sigma
        )
        raw_metrics[target] = metrics
        per_target[target] = {"unit": _unit_for_target(target), **metrics}
        for interval in per_target[target]["intervals"].values():
            interval["unit"] = _unit_for_target(target)

    result: Dict[str, Any] = {"per_target": per_target}
    if clock_col in raw_metrics:
        result["clock_interval_widths"] = {
            "seconds": _scale_interval_widths(raw_metrics[clock_col], 1.0, "s"),
            "nanoseconds": _scale_interval_widths(raw_metrics[clock_col], 1e9, "ns"),
            "range_equivalent_metres": {
                **_scale_interval_widths(
                    raw_metrics[clock_col], SPEED_OF_LIGHT_M_S, "m"
                ),
                "conversion": "c * dt",
                "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
            },
        }
    return result


def _student_t_score_block(
    actual: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    degrees_of_freedom: np.ndarray,
    target_cols: Sequence[str],
    clock_col: str,
    coverages: Sequence[float],
    min_scale: float,
) -> Dict[str, Any]:
    per_target: Dict[str, Any] = {}
    raw_metrics: Dict[str, Dict[str, Any]] = {}
    for index, target in enumerate(target_cols):
        if target in DERIVED_3D_TARGETS:
            continue
        metrics = _student_t_target_metrics(
            actual[..., index],
            mean[..., index],
            scale[..., index],
            degrees_of_freedom[..., index],
            coverages,
            min_scale,
        )
        raw_metrics[target] = metrics
        per_target[target] = {"unit": _unit_for_target(target), **metrics}
        for interval in per_target[target]["intervals"].values():
            interval["unit"] = _unit_for_target(target)

    result: Dict[str, Any] = {"per_target": per_target}
    if clock_col in raw_metrics:
        result["clock_interval_widths"] = {
            "seconds": _scale_interval_widths(raw_metrics[clock_col], 1.0, "s"),
            "nanoseconds": _scale_interval_widths(raw_metrics[clock_col], 1e9, "ns"),
            "range_equivalent_metres": {
                **_scale_interval_widths(
                    raw_metrics[clock_col], SPEED_OF_LIGHT_M_S, "m"
                ),
                "conversion": "c * dt",
                "speed_of_light_m_s": SPEED_OF_LIGHT_M_S,
            },
        }
    return result


def _validate_degrees_of_freedom(
    degrees_of_freedom: Optional[np.ndarray | float],
    target_shape: Tuple[int, ...],
) -> np.ndarray:
    if degrees_of_freedom is None:
        raise ValueError("df is required when distribution='student_t'")
    values = np.asarray(degrees_of_freedom, dtype=np.float64)
    try:
        values = np.broadcast_to(values, target_shape)
    except ValueError as error:
        raise ValueError(
            f"df must be scalar or broadcastable to forecast shape {target_shape}; "
            f"got {values.shape}"
        ) from error
    invalid = np.isfinite(values) & (values <= 0)
    if np.any(invalid):
        raise ValueError("all finite Student-t degrees of freedom must be positive")
    return values


def compute_probabilistic_metrics(
    actual: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    target_cols: Sequence[str],
    horizons: Optional[Mapping[str, int]] = HORIZON_MAP,
    coverages: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
    clock_col: str = "Error_Clock",
    min_sigma: float = 1e-12,
    distribution: str = "gaussian",
    df: Optional[np.ndarray | float] = None,
    valid_mask: Optional[np.ndarray] = None,
    satellite_ids: Optional[Sequence[Any]] = None,
    constellations: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Evaluate Gaussian or Student-t NLL and central interval calibration.

    For ``distribution='student_t'``, ``sigma`` is the Student-t scale (not its
    standard deviation) and ``df`` must be a positive scalar or an array
    broadcastable to the forecast tensor.
    """

    actual_tensor, mean_tensor, columns = _validate_forecasts(actual, mean, target_cols)
    actual_tensor = _mask_actual(actual_tensor, valid_mask)
    sigma_tensor = _validate_sigma(actual_tensor, sigma)
    if min_sigma <= 0:
        raise ValueError("min_sigma must be positive")
    distribution_name = distribution.lower().replace("-", "_")
    if distribution_name in {"normal", "gaussian"}:
        distribution_name = "gaussian"
        degrees_of_freedom = None
        score_block = lambda a, m, s, _df: _gaussian_score_block(
            a, m, s, columns, clock_col, coverages, min_sigma
        )
    elif distribution_name in {"student", "student_t", "studentt"}:
        distribution_name = "student_t"
        degrees_of_freedom = _validate_degrees_of_freedom(df, actual_tensor.shape)
        score_block = lambda a, m, s, selected_df: _student_t_score_block(
            a,
            m,
            s,
            selected_df,
            columns,
            clock_col,
            coverages,
            min_sigma,
        )
    else:
        raise ValueError("distribution must be 'gaussian' or 'student_t'")

    report: Dict[str, Any] = {
        "protocol": {
            "distribution": f"independent {distribution_name} per target and lead",
            "sigma_definition": (
                "predictive standard deviation in each target's native unit"
                if distribution_name == "gaussian"
                else "Student-t scale in each target's native unit"
            ),
            "unit_policy": "NLL and interval metrics are reported per target only",
        },
        "all_forecast_points": score_block(
            actual_tensor, mean_tensor, sigma_tensor, degrees_of_freedom
        ),
        "horizons": {},
    }
    for label, step in _normalise_horizons(horizons, actual_tensor.shape[1]).items():
        if step > actual_tensor.shape[1]:
            report["horizons"][label] = {
                "requested_lead_step": step,
                "available": False,
                "exact_lead": None,
                "cumulative_window_1_to_lead": None,
            }
            continue
        report["horizons"][label] = {
            "requested_lead_step": step,
            "available": True,
            "exact_lead": score_block(
                actual_tensor[:, step - 1 : step],
                mean_tensor[:, step - 1 : step],
                sigma_tensor[:, step - 1 : step],
                None
                if degrees_of_freedom is None
                else degrees_of_freedom[:, step - 1 : step],
            ),
            "cumulative_window_1_to_lead": score_block(
                actual_tensor[:, :step],
                mean_tensor[:, :step],
                sigma_tensor[:, :step],
                None if degrees_of_freedom is None else degrees_of_freedom[:, :step],
            ),
        }

    sat_labels = _normalise_slice_labels(satellite_ids, actual_tensor.shape[0], "satellite_ids")
    constellation_labels = _normalise_slice_labels(
        constellations, actual_tensor.shape[0], "constellations"
    )
    if constellation_labels is not None and sat_labels is None:
        raise ValueError("constellations can only be supplied together with satellite_ids")
    if sat_labels is not None:
        if constellation_labels is None:
            constellation_labels = np.asarray(
                [infer_constellation(satellite) for satellite in sat_labels], dtype=object
            )
        slices: Dict[str, Dict[str, Any]] = {"per_satellite": {}, "per_constellation": {}}
        for satellite in sorted(set(sat_labels.tolist())):
            mask = sat_labels == satellite
            slices["per_satellite"][satellite] = compute_probabilistic_metrics(
                actual_tensor[mask],
                mean_tensor[mask],
                sigma_tensor[mask],
                columns,
                horizons=horizons,
                coverages=coverages,
                clock_col=clock_col,
                min_sigma=min_sigma,
                distribution=distribution_name,
                df=(None if degrees_of_freedom is None else degrees_of_freedom[mask]),
            )
        for constellation in sorted(set(constellation_labels.tolist())):
            mask = constellation_labels == constellation
            slices["per_constellation"][str(constellation)] = compute_probabilistic_metrics(
                actual_tensor[mask],
                mean_tensor[mask],
                sigma_tensor[mask],
                columns,
                horizons=horizons,
                coverages=coverages,
                clock_col=clock_col,
                min_sigma=min_sigma,
                distribution=distribution_name,
                df=(None if degrees_of_freedom is None else degrees_of_freedom[mask]),
            )
        report["slices"] = slices
    return report


def compute_student_t_metrics(
    actual: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    df: np.ndarray | float,
    target_cols: Sequence[str],
    horizons: Optional[Mapping[str, int]] = HORIZON_MAP,
    coverages: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
    clock_col: str = "Error_Clock",
    min_scale: float = 1e-12,
    valid_mask: Optional[np.ndarray] = None,
    satellite_ids: Optional[Sequence[Any]] = None,
    constellations: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper for Student-t probabilistic evaluation."""

    return compute_probabilistic_metrics(
        actual=actual,
        mean=mean,
        sigma=scale,
        target_cols=target_cols,
        horizons=horizons,
        coverages=coverages,
        clock_col=clock_col,
        min_sigma=min_scale,
        distribution="student_t",
        df=df,
        valid_mask=valid_mask,
        satellite_ids=satellite_ids,
        constellations=constellations,
    )


def _as_sample_tensor(samples: np.ndarray, actual_shape: Tuple[int, ...]) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    # One-series convenience form: (draw, horizon, target).
    if values.ndim == 3 and actual_shape[0] == 1:
        values = values[:, None, :, :]
    if values.ndim != 4:
        raise ValueError(
            "samples must have shape (draw, series, horizon, target), or "
            f"(draw, horizon, target) for one series; got {values.shape}"
        )
    if tuple(values.shape[1:]) != tuple(actual_shape):
        raise ValueError(
            f"sample event shape must be {actual_shape}; got {values.shape[1:]}"
        )
    if values.shape[0] < 2:
        raise ValueError("at least two forecast samples are required")
    return values


def _empirical_crps(actual: np.ndarray, samples: np.ndarray) -> Tuple[Optional[float], int]:
    """Mean empirical CRPS using an O(S log S) sorted-sample identity."""

    target = np.asarray(actual, dtype=np.float64).reshape(-1)
    draws = np.asarray(samples, dtype=np.float64).reshape(samples.shape[0], -1)
    valid = np.isfinite(target) & np.all(np.isfinite(draws), axis=0)
    if not np.any(valid):
        return None, 0
    target = target[valid]
    draws = draws[:, valid]
    first_term = np.mean(np.abs(draws - target[None, :]), axis=0)
    ordered = np.sort(draws, axis=0)
    sample_count = ordered.shape[0]
    weights = 2.0 * np.arange(sample_count, dtype=np.float64) - sample_count + 1.0
    half_pairwise_term = np.sum(weights[:, None] * ordered, axis=0) / sample_count**2
    values = first_term - half_pairwise_term
    return float(np.mean(values)), int(values.size)


def _energy_score_xyz(actual_xyz: np.ndarray, sample_xyz: np.ndarray) -> Tuple[Optional[float], int]:
    target = np.asarray(actual_xyz, dtype=np.float64).reshape(-1, 3)
    draws = np.asarray(sample_xyz, dtype=np.float64).reshape(sample_xyz.shape[0], -1, 3)
    valid = np.all(np.isfinite(target), axis=-1) & np.all(
        np.all(np.isfinite(draws), axis=-1), axis=0
    )
    if not np.any(valid):
        return None, 0
    target = target[valid]
    draws = draws[:, valid]
    first_term = np.linalg.norm(draws - target[None, ...], axis=-1).mean(axis=0)
    pairwise_sum = np.zeros(target.shape[0], dtype=np.float64)
    for left in range(draws.shape[0]):
        for right in range(draws.shape[0]):
            pairwise_sum += np.linalg.norm(draws[left] - draws[right], axis=-1)
    second_term = 0.5 * pairwise_sum / draws.shape[0] ** 2
    return float(np.mean(first_term - second_term)), int(target.shape[0])


def _sample_score_block(
    actual: np.ndarray,
    samples: np.ndarray,
    target_cols: Sequence[str],
    clock_col: str,
) -> Dict[str, Any]:
    per_target: Dict[str, Any] = {}
    for index, target in enumerate(target_cols):
        if target in DERIVED_3D_TARGETS:
            continue
        crps, count = _empirical_crps(actual[..., index], samples[..., index])
        per_target[target] = {
            "unit": _unit_for_target(target),
            "count": count,
            "crps": crps,
        }

    report: Dict[str, Any] = {"per_target": per_target}
    xyz_indices = _xyz_indices(target_cols)
    if xyz_indices is not None:
        energy, count = _energy_score_xyz(
            actual[..., list(xyz_indices)], samples[..., list(xyz_indices)]
        )
        report["orbit_xyz_energy_score"] = {
            "unit": "m",
            "count": count,
            "energy_score": energy,
        }
    if clock_col in per_target:
        clock_crps = per_target[clock_col]["crps"]
        count = per_target[clock_col]["count"]
        report["clock_crps"] = {
            "seconds": {"unit": "s", "count": count, "crps": clock_crps},
            "nanoseconds": {
                "unit": "ns",
                "count": count,
                "crps": None if clock_crps is None else float(clock_crps * 1e9),
            },
            "range_equivalent_metres": {
                "unit": "m",
                "count": count,
                "conversion": "c * dt",
                "crps": (
                    None
                    if clock_crps is None
                    else float(clock_crps * SPEED_OF_LIGHT_M_S)
                ),
            },
        }
    return report


def compute_sample_metrics(
    actual: np.ndarray,
    samples: np.ndarray,
    target_cols: Sequence[str],
    horizons: Optional[Mapping[str, int]] = HORIZON_MAP,
    clock_col: str = "Error_Clock",
    valid_mask: Optional[np.ndarray] = None,
    satellite_ids: Optional[Sequence[Any]] = None,
    constellations: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Compute empirical CRPS per target and an XYZ-only energy score."""

    actual_tensor, _ = _as_forecast_tensor(actual, "actual")
    actual_tensor = _mask_actual(actual_tensor, valid_mask)
    columns = list(target_cols)
    if len(columns) != actual_tensor.shape[-1]:
        raise ValueError("target_cols does not match the sample target dimension")
    sample_tensor = _as_sample_tensor(samples, actual_tensor.shape)
    report: Dict[str, Any] = {
        "protocol": {
            "sample_axis": 0,
            "sample_shape": "draw, series, forecast_lead, target",
            "crps": "empirical univariate CRPS per target",
            "energy_score": "multivariate energy score over XYZ only; clock is excluded",
            "unit_policy": "no score combines orbit metres with clock seconds",
        },
        "all_forecast_points": _sample_score_block(
            actual_tensor, sample_tensor, columns, clock_col
        ),
        "horizons": {},
    }
    for label, step in _normalise_horizons(horizons, actual_tensor.shape[1]).items():
        if step > actual_tensor.shape[1]:
            report["horizons"][label] = {
                "requested_lead_step": step,
                "available": False,
                "exact_lead": None,
                "cumulative_window_1_to_lead": None,
            }
            continue
        report["horizons"][label] = {
            "requested_lead_step": step,
            "available": True,
            "exact_lead": _sample_score_block(
                actual_tensor[:, step - 1 : step],
                sample_tensor[:, :, step - 1 : step],
                columns,
                clock_col,
            ),
            "cumulative_window_1_to_lead": _sample_score_block(
                actual_tensor[:, :step], sample_tensor[:, :, :step], columns, clock_col
            ),
        }

    sat_labels = _normalise_slice_labels(satellite_ids, actual_tensor.shape[0], "satellite_ids")
    constellation_labels = _normalise_slice_labels(
        constellations, actual_tensor.shape[0], "constellations"
    )
    if constellation_labels is not None and sat_labels is None:
        raise ValueError("constellations can only be supplied together with satellite_ids")
    if sat_labels is not None:
        if constellation_labels is None:
            constellation_labels = np.asarray(
                [infer_constellation(satellite) for satellite in sat_labels], dtype=object
            )
        slices: Dict[str, Dict[str, Any]] = {"per_satellite": {}, "per_constellation": {}}
        for satellite in sorted(set(sat_labels.tolist())):
            mask = sat_labels == satellite
            slices["per_satellite"][satellite] = compute_sample_metrics(
                actual_tensor[mask],
                sample_tensor[:, mask],
                columns,
                horizons=horizons,
                clock_col=clock_col,
            )
        for constellation in sorted(set(constellation_labels.tolist())):
            mask = constellation_labels == constellation
            slices["per_constellation"][str(constellation)] = compute_sample_metrics(
                actual_tensor[mask],
                sample_tensor[:, mask],
                columns,
                horizons=horizons,
                clock_col=clock_col,
            )
        report["slices"] = slices
    return report


def _metric_at_path(metrics: Mapping[str, Any], path: str) -> float:
    current: Any = metrics
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(path)
        current = current[component]
    if isinstance(current, bool) or not isinstance(current, (int, float, np.number)):
        raise TypeError(f"metric {path!r} is not numeric")
    value = float(current)
    if not math.isfinite(value):
        raise ValueError(f"metric {path!r} is not finite")
    return value


def compute_skill_score(candidate_error: float, baseline_error: float) -> float:
    """Return error skill ``1 - candidate / baseline`` (positive is better).

    This definition is appropriate for non-negative loss/error metrics such as
    MAE, RMSE, CRPS, and energy score. A zero or negative baseline does not
    define a meaningful ratio and is rejected.
    """

    candidate_value = float(candidate_error)
    baseline_value = float(baseline_error)
    if not math.isfinite(candidate_value) or not math.isfinite(baseline_value):
        raise ValueError("candidate and baseline errors must be finite")
    if candidate_value < 0 or baseline_value <= 0:
        raise ValueError("skill score requires candidate >= 0 and baseline > 0")
    return float(1.0 - candidate_value / baseline_value)


def _normalise_rule(metric: str, config: Any) -> PromotionRule:
    if isinstance(config, PromotionRule):
        return config if config.metric == metric else PromotionRule(metric=metric, **{
            key: value for key, value in asdict(config).items() if key != "metric"
        })
    if isinstance(config, str):
        return PromotionRule(metric=metric, direction=config)
    if config is None:
        return PromotionRule(metric=metric)
    if not isinstance(config, Mapping):
        raise TypeError(f"promotion rule for {metric!r} must be a mapping, string, or PromotionRule")
    values = dict(config)
    values.pop("metric", None)
    if "min_improvement" in values and "min_absolute_improvement" not in values:
        values["min_absolute_improvement"] = values.pop("min_improvement")
    if "min_relative" in values and "min_relative_improvement" not in values:
        values["min_relative_improvement"] = values.pop("min_relative")
    return PromotionRule(metric=metric, **values)


def _normalise_rules(
    rules: Mapping[str, Any] | Sequence[PromotionRule | Mapping[str, Any]],
) -> List[PromotionRule]:
    if isinstance(rules, Mapping):
        return [_normalise_rule(str(metric), config) for metric, config in rules.items()]
    normalised: List[PromotionRule] = []
    for config in rules:
        if isinstance(config, PromotionRule):
            normalised.append(config)
        elif isinstance(config, Mapping) and "metric" in config:
            values = dict(config)
            metric = str(values.pop("metric"))
            normalised.append(_normalise_rule(metric, values))
        else:
            raise TypeError("sequence promotion rules must include a metric path")
    return normalised


def compare_candidate_to_baseline(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    rules: Mapping[str, Any] | Sequence[PromotionRule | Mapping[str, Any]],
    require_all: bool = True,
) -> Dict[str, Any]:
    """Compare configured metric paths and return a fail-closed promotion report."""

    normalised_rules = _normalise_rules(rules)
    if not normalised_rules:
        raise ValueError("at least one promotion rule is required")
    comparisons: List[Dict[str, Any]] = []
    for rule in normalised_rules:
        direction = rule.direction.lower()
        if direction in {"min", "minimize", "lower_is_better"}:
            direction = "lower"
        elif direction in {"max", "maximize", "higher_is_better"}:
            direction = "higher"
        if direction not in {"lower", "higher"}:
            raise ValueError(f"unsupported direction {rule.direction!r} for {rule.metric!r}")
        if rule.min_absolute_improvement < 0 or rule.min_relative_improvement < 0:
            raise ValueError("minimum improvements cannot be negative")

        comparison: Dict[str, Any] = {"metric": rule.metric, "direction": direction}
        try:
            candidate = _metric_at_path(candidate_metrics, rule.metric)
            baseline = _metric_at_path(baseline_metrics, rule.metric)
        except (KeyError, TypeError, ValueError) as error:
            comparison.update({"passed": False, "reason": str(error)})
            comparisons.append(comparison)
            continue

        absolute_improvement = baseline - candidate if direction == "lower" else candidate - baseline
        if baseline == 0:
            relative_improvement = math.inf if absolute_improvement > 0 else 0.0
        else:
            relative_improvement = absolute_improvement / abs(baseline)
        strictly_better = absolute_improvement > 0.0
        equal_allowed = rule.allow_equal and absolute_improvement == 0.0
        thresholds_met = (
            absolute_improvement >= rule.min_absolute_improvement
            and relative_improvement >= rule.min_relative_improvement
        )
        passed = bool((strictly_better or equal_allowed) and thresholds_met)
        comparison.update(
            {
                "candidate": candidate,
                "baseline": baseline,
                "absolute_improvement": float(absolute_improvement),
                "relative_improvement": float(relative_improvement),
                "skill_score": (
                    float(1.0 - candidate / baseline)
                    if direction == "lower" and candidate >= 0 and baseline > 0
                    else None
                ),
                "min_absolute_improvement": rule.min_absolute_improvement,
                "min_relative_improvement": rule.min_relative_improvement,
                "allow_equal": rule.allow_equal,
                "passed": passed,
                "reason": "passed" if passed else "candidate did not clear the configured baseline gate",
            }
        )
        comparisons.append(comparison)

    passed_count = sum(bool(item["passed"]) for item in comparisons)
    overall_passed = passed_count == len(comparisons) if require_all else passed_count > 0
    return {
        "passed": overall_passed,
        "require_all": require_all,
        "passed_rules": passed_count,
        "total_rules": len(comparisons),
        "comparisons": comparisons,
        "failures": [item for item in comparisons if not item["passed"]],
    }


def assert_candidate_beats_baseline(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    rules: Mapping[str, Any] | Sequence[PromotionRule | Mapping[str, Any]],
    require_all: bool = True,
) -> Dict[str, Any]:
    """Return the comparison report or raise :class:`PromotionGateError`."""

    report = compare_candidate_to_baseline(
        candidate_metrics, baseline_metrics, rules, require_all=require_all
    )
    if not report["passed"]:
        failed_paths = [failure["metric"] for failure in report["failures"]]
        raise PromotionGateError(
            "Candidate failed baseline promotion gates: " + ", ".join(failed_paths)
        )
    return report


# Clear aliases for callers that use promotion-oriented terminology.
assess_promotion = compare_candidate_to_baseline
assert_promotion_ready = assert_candidate_beats_baseline


def _stack_satellite_forecasts(
    all_actuals: Mapping[str, np.ndarray],
    all_preds: Mapping[str, np.ndarray],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    satellites = sorted(set(all_actuals) & set(all_preds))
    if not satellites:
        raise ValueError("actual and prediction dictionaries have no common satellites")
    missing_actual = sorted(set(all_preds) - set(all_actuals))
    missing_predictions = sorted(set(all_actuals) - set(all_preds))
    if missing_actual or missing_predictions:
        raise ValueError(
            f"satellite key mismatch: missing actual={missing_actual}, "
            f"missing predictions={missing_predictions}"
        )
    actual = np.stack([np.asarray(all_actuals[satellite]) for satellite in satellites])
    predicted = np.stack([np.asarray(all_preds[satellite]) for satellite in satellites])
    return satellites, actual, predicted


def _legacy_summary(summary: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "MAE": float("nan") if summary["mae"] is None else float(summary["mae"]),
        "RMSE": float("nan") if summary["rmse"] is None else float(summary["rmse"]),
        "MedianAE": (
            float("nan") if summary["median_ae"] is None else float(summary["median_ae"])
        ),
        "P90_AE": float("nan") if summary["p90_ae"] is None else float(summary["p90_ae"]),
        "P95_AE": float("nan") if summary["p95_ae"] is None else float(summary["p95_ae"]),
        "P99_AE": float("nan") if summary["p99_ae"] is None else float(summary["p99_ae"]),
    }


def _legacy_target_summary(block: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    if target in DERIVED_3D_TARGETS and "orbit_3d_vector_error" in block:
        return block["orbit_3d_vector_error"]
    return block["per_target"][target]


def compute_aggregate_metrics(
    all_actuals: Dict[str, np.ndarray],
    all_preds: Dict[str, np.ndarray],
    target_cols: List[str] = TARGET_COLS_5,
) -> Tuple[Dict, Dict]:
    """Compatibility adapter for the original dictionary-based evaluator.

    Unlike the historical implementation, ``3D_Orbit_Error`` is scored from
    XYZ residuals and additional tail metrics are included. Existing MAE/RMSE
    keys and aggregate shapes remain intact.
    """

    satellites, actual, predicted = _stack_satellite_forecasts(all_actuals, all_preds)
    report = evaluate_forecasts(
        actual,
        predicted,
        target_cols,
        horizons={},
        satellite_ids=satellites,
    )
    per_sat_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for satellite in satellites:
        block = report["slices"]["per_satellite"][satellite]["all_forecast_points"]
        per_sat_results[satellite] = {
            target: _legacy_summary(_legacy_target_summary(block, target))
            for target in target_cols
        }

    aggregate = {
        target: {
            "Mean_MAE": float(
                np.mean([per_sat_results[satellite][target]["MAE"] for satellite in satellites])
            ),
            "Mean_RMSE": float(
                np.mean([per_sat_results[satellite][target]["RMSE"] for satellite in satellites])
            ),
            "Mean_MedianAE": float(
                np.mean(
                    [per_sat_results[satellite][target]["MedianAE"] for satellite in satellites]
                )
            ),
            "Mean_P90_AE": float(
                np.mean([per_sat_results[satellite][target]["P90_AE"] for satellite in satellites])
            ),
            "Mean_P95_AE": float(
                np.mean([per_sat_results[satellite][target]["P95_AE"] for satellite in satellites])
            ),
            "Mean_P99_AE": float(
                np.mean([per_sat_results[satellite][target]["P99_AE"] for satellite in satellites])
            ),
        }
        for target in target_cols
    }
    return per_sat_results, aggregate


def compute_multi_horizon_metrics(
    all_actuals: Dict[str, np.ndarray],
    all_preds: Dict[str, np.ndarray],
    target_cols: List[str] = TARGET_COLS_5,
    horizons: Dict[str, int] = HORIZON_MAP,
) -> Dict[str, Dict[str, float]]:
    """Compatibility adapter returning historical cumulative-window MAE keys.

    New code should consume :func:`evaluate_forecasts`, whose horizon fields
    name both exact-lead and cumulative-window semantics explicitly.
    """

    _, actual, predicted = _stack_satellite_forecasts(all_actuals, all_preds)
    report = evaluate_forecasts(actual, predicted, target_cols, horizons=horizons)
    results: Dict[str, Dict[str, float]] = {}
    for label in horizons:
        horizon_report = report["horizons"][label]
        if not horizon_report["available"]:
            results[label] = {target: float("nan") for target in target_cols}
            continue
        block = horizon_report["cumulative_window_1_to_lead"]
        results[label] = {
            target: float(_legacy_target_summary(block, target)["mae"])
            for target in target_cols
        }
    return results


def compute_tensor_horizon_metrics(
    actual_real: np.ndarray,
    pred_real: np.ndarray,
    target_cols: List[str],
    horizons: Dict[str, int] = HORIZON_MAP,
) -> Dict[str, Dict[str, Any]]:
    """Compatibility adapter with legacy flat exact-lead keys plus explicit views.

    The invalid historical ``Overall_MAE``/``Overall_RMSE`` fields are omitted
    because they mixed orbit metres and clock seconds.
    """

    report = evaluate_forecasts(actual_real, pred_real, target_cols, horizons=horizons)
    results: Dict[str, Dict[str, Any]] = {}
    for label in horizons:
        horizon_report = report["horizons"][label]
        legacy: Dict[str, Any] = {
            "requested_lead_step": horizon_report["requested_lead_step"],
            "available": horizon_report["available"],
            "exact_lead": horizon_report["exact_lead"],
            "cumulative_window_1_to_lead": horizon_report["cumulative_window_1_to_lead"],
        }
        if not horizon_report["available"]:
            for target in target_cols:
                legacy[f"{target}_MAE"] = float("nan")
                legacy[f"{target}_RMSE"] = float("nan")
        else:
            block = horizon_report["exact_lead"]
            for target in target_cols:
                summary = _legacy_target_summary(block, target)
                legacy[f"{target}_MAE"] = float(summary["mae"])
                legacy[f"{target}_RMSE"] = float(summary["rmse"])
                legacy[f"{target}_MedianAE"] = float(summary["median_ae"])
                legacy[f"{target}_P90_AE"] = float(summary["p90_ae"])
                legacy[f"{target}_P95_AE"] = float(summary["p95_ae"])
                legacy[f"{target}_P99_AE"] = float(summary["p99_ae"])
        results[label] = legacy
    return results


def print_metrics_summary(
    aggregate: Dict,
    horizon_results: Dict,
    target_cols: List[str] = TARGET_COLS_5,
) -> None:
    """Print the compatibility summary without combining unlike units."""

    print("\n" + "=" * 76)
    print("EVALUATION METRICS SUMMARY (UNIT-AWARE; NO CROSS-UNIT OVERALL SCORE)")
    print("=" * 76)
    print(f"  {'Target':<22}  {'Unit':>6}  {'Mean MAE':>14}  {'Mean RMSE':>14}")
    print("  " + "-" * 66)
    for target in target_cols:
        print(
            f"  {target:<22}  {_unit_for_target(target):>6}  "
            f"{aggregate[target]['Mean_MAE']:>14.6f}  "
            f"{aggregate[target]['Mean_RMSE']:>14.6f}"
        )

    print("\n" + "=" * 80)
    print("MULTI-HORIZON MAE BREAKDOWN (LEGACY ADAPTER)")
    print("=" * 80)
    print(f"  {'Horizon':<12}", end="")
    for target in target_cols:
        short = target.replace("Error_", "").replace("3D_Orbit_", "3D_")
        print(f"  {short:>12}", end="")
    print()
    print("  " + "-" * 80)
    for label, metrics in horizon_results.items():
        print(f"  {label:<12}", end="")
        for target in target_cols:
            value = metrics.get(target, metrics.get(f"{target}_MAE", float("nan")))
            print(f"  {value:>12.6f}", end="")
        print()


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


def save_metrics_summary(
    filepath: str,
    model_name: str,
    aggregate: Dict,
    horizon_results: Dict,
    per_sat_results: Optional[Dict] = None,
    extra_meta: Optional[Dict] = None,
    evaluation_report: Optional[Dict] = None,
) -> None:
    """Save a JSON-safe compatibility summary and optional canonical report."""

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    summary: Dict[str, Any] = {
        "model": model_name,
        "aggregate_metrics": aggregate,
        "multi_horizon_metrics": horizon_results,
    }
    if per_sat_results is not None:
        summary["per_satellite_metrics"] = per_sat_results
    if extra_meta is not None:
        summary["metadata"] = extra_meta
    if evaluation_report is not None:
        summary["unit_aware_evaluation"] = evaluation_report

    with open(filepath, "w", encoding="utf-8") as file_handle:
        json.dump(_json_safe(summary), file_handle, indent=2, allow_nan=False)
    print(f"\n  Metrics saved to: {filepath}")

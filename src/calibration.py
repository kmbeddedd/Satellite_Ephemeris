"""Time-aware split-conformal calibration for multi-horizon forecasts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _finite_scores(
    actual: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    if actual.shape != mean.shape or actual.shape != scale.shape or actual.ndim != 3:
        raise ValueError("actual, mean, and scale must share shape (series, horizon, target)")
    valid = np.isfinite(actual) & np.isfinite(mean) & np.isfinite(scale) & (scale > 0)
    if mask is not None:
        if np.asarray(mask).shape != actual.shape:
            raise ValueError("mask must match actual")
        valid &= np.asarray(mask, dtype=bool)
    scores = np.full(actual.shape, np.nan, dtype=np.float64)
    scores[valid] = np.abs(actual[valid] - mean[valid]) / scale[valid]
    return scores, valid


def _conformal_quantile(values: np.ndarray, coverage: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    level = min(1.0, np.ceil((finite.size + 1) * coverage) / finite.size)
    return float(np.quantile(finite, level, method="higher"))


def fit_scaled_conformal(
    actual: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray | None = None,
    coverages: Sequence[float] = (0.8, 0.9, 0.95),
    min_cell_samples: int = 20,
) -> dict[str, Any]:
    """Fit per-lead/per-target multipliers on a chronological calibration fold.

    Sparse lead/target cells fall back to a target-wide calibration score. The
    caller is responsible for ensuring this fold precedes the final test period.
    """
    if min_cell_samples < 1:
        raise ValueError("min_cell_samples must be positive")
    scores, valid = _finite_scores(actual, mean, scale, mask)
    horizon, targets = scores.shape[1:]
    result: dict[str, Any] = {
        "method": "scaled split conformal",
        "calibration_shape": list(scores.shape),
        "min_cell_samples": int(min_cell_samples),
        "coverages": {},
    }
    for coverage in coverages:
        coverage = float(coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverages must lie strictly between zero and one")
        multipliers = np.empty((horizon, targets), dtype=np.float64)
        counts = valid.sum(axis=0).astype(np.int64)
        fallback_used = np.zeros((horizon, targets), dtype=bool)
        for target in range(targets):
            fallback = _conformal_quantile(scores[:, :, target], coverage)
            for lead in range(horizon):
                if counts[lead, target] >= min_cell_samples:
                    multipliers[lead, target] = _conformal_quantile(
                        scores[:, lead, target], coverage
                    )
                else:
                    multipliers[lead, target] = fallback
                    fallback_used[lead, target] = True
        result["coverages"][f"{coverage:.6g}"] = {
            "coverage": coverage,
            "multipliers": multipliers.tolist(),
            "counts": counts.tolist(),
            "target_fallback_used": fallback_used.tolist(),
        }
    return result


def conformal_interval(
    mean: np.ndarray,
    scale: np.ndarray,
    calibration: dict[str, Any],
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    key = f"{float(coverage):.6g}"
    if key not in calibration["coverages"]:
        raise KeyError(f"Coverage {coverage} was not fitted")
    mean = np.asarray(mean, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    multiplier = np.asarray(calibration["coverages"][key]["multipliers"], dtype=np.float64)
    if mean.shape != scale.shape or mean.shape[1:] != multiplier.shape:
        raise ValueError("mean/scale shape is incompatible with fitted calibration")
    half_width = scale * multiplier[None, :, :]
    return mean - half_width, mean + half_width


def evaluate_conformal_intervals(
    actual: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    calibration: dict[str, Any],
    target_cols: Sequence[str],
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    valid_base = np.isfinite(actual)
    if mask is not None:
        valid_base &= np.asarray(mask, dtype=bool)
    report: dict[str, Any] = {"per_target": {}}
    for target_index, target in enumerate(target_cols):
        target_report = {}
        for entry in calibration["coverages"].values():
            coverage = float(entry["coverage"])
            lower, upper = conformal_interval(mean, scale, calibration, coverage)
            valid = (
                valid_base[..., target_index]
                & np.isfinite(lower[..., target_index])
                & np.isfinite(upper[..., target_index])
            )
            if np.any(valid):
                contained = (
                    actual[..., target_index] >= lower[..., target_index]
                ) & (actual[..., target_index] <= upper[..., target_index])
                width = upper[..., target_index] - lower[..., target_index]
                target_report[f"{coverage:.6g}"] = {
                    "nominal_coverage": coverage,
                    "empirical_coverage": float(np.mean(contained[valid])),
                    "mean_width": float(np.mean(width[valid])),
                    "count": int(valid.sum()),
                }
            else:
                target_report[f"{coverage:.6g}"] = {
                    "nominal_coverage": coverage,
                    "empirical_coverage": None,
                    "mean_width": None,
                    "count": 0,
                }
        report["per_target"][str(target)] = target_report
    return report


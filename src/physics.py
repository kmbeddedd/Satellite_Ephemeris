"""Coordinate transforms for physics-residual GNSS forecasting."""

from __future__ import annotations

import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0


def _unit(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(norm <= eps):
        raise ValueError("Cannot construct an orbital frame from a zero-length vector")
    return vector / norm


def ric_basis(position_ecef: np.ndarray, velocity_ecef: np.ndarray) -> np.ndarray:
    """Return ECEF basis vectors for radial, in-track, and cross-track axes.

    The returned array has shape ``(..., 3, 3)`` and stores R/I/C basis
    vectors as columns. Inputs and outputs use the same distance unit.
    """
    position = np.asarray(position_ecef, dtype=np.float64)
    velocity = np.asarray(velocity_ecef, dtype=np.float64)
    if position.shape != velocity.shape or position.shape[-1] != 3:
        raise ValueError("position_ecef and velocity_ecef must share shape (..., 3)")
    radial = _unit(position)
    cross_track = _unit(np.cross(position, velocity))
    in_track = _unit(np.cross(cross_track, radial))
    return np.stack([radial, in_track, cross_track], axis=-1)


def ecef_error_to_ric(
    error_ecef: np.ndarray,
    position_ecef: np.ndarray,
    velocity_ecef: np.ndarray,
) -> np.ndarray:
    """Rotate an ECEF error vector into the local orbital R/I/C frame."""
    error = np.asarray(error_ecef, dtype=np.float64)
    basis = ric_basis(position_ecef, velocity_ecef)
    if error.shape != basis.shape[:-1]:
        raise ValueError("error_ecef must have the same (..., 3) shape as position")
    return np.einsum("...ji,...j->...i", basis, error)


def ric_error_to_ecef(
    error_ric: np.ndarray,
    position_ecef: np.ndarray,
    velocity_ecef: np.ndarray,
) -> np.ndarray:
    """Rotate a local orbital R/I/C error vector back to ECEF."""
    error = np.asarray(error_ric, dtype=np.float64)
    basis = ric_basis(position_ecef, velocity_ecef)
    if error.shape != basis.shape[:-1]:
        raise ValueError("error_ric must have the same (..., 3) shape as position")
    return np.einsum("...ij,...j->...i", basis, error)


def clock_seconds_to_metres(clock_seconds: np.ndarray) -> np.ndarray:
    return np.asarray(clock_seconds, dtype=np.float64) * SPEED_OF_LIGHT_M_S


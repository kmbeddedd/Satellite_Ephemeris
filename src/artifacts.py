"""Reproducibility and model-artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ARTIFACT_SCHEMA_VERSION = 1


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed supported runtimes and select deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_sha(cwd: str | os.PathLike[str] | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def scaler_to_state(scaler: StandardScaler) -> dict[str, Any]:
    """Serialize the fitted state needed to reproduce a StandardScaler."""
    state: dict[str, Any] = {
        "class": type(scaler).__name__,
        "with_mean": scaler.with_mean,
        "with_std": scaler.with_std,
    }
    for name in ("mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_"):
        if hasattr(scaler, name):
            value = getattr(scaler, name)
            state[name] = value.tolist() if isinstance(value, np.ndarray) else int(value)
    if hasattr(scaler, "feature_names_in_"):
        state["feature_names_in_"] = scaler.feature_names_in_.tolist()
    return state


def scaler_from_state(state: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler(
        with_mean=bool(state.get("with_mean", True)),
        with_std=bool(state.get("with_std", True)),
    )
    for name in ("mean_", "scale_", "var_"):
        if name in state:
            setattr(scaler, name, np.asarray(state[name], dtype=np.float64))
    scaler.n_features_in_ = int(state["n_features_in_"])
    samples_seen = state.get("n_samples_seen_", 1)
    scaler.n_samples_seen_ = np.asarray(samples_seen) if isinstance(samples_seen, list) else samples_seen
    if "feature_names_in_" in state:
        scaler.feature_names_in_ = np.asarray(state["feature_names_in_"], dtype=object)
    return scaler


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    """Write a human-readable sidecar manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)


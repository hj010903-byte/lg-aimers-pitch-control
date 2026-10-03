"""Probability metrics used by the project."""

from __future__ import annotations

import numpy as np


def _as_vector(values: object, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def brier_score(y_true: object, y_prob: object) -> float:
    """Return mean squared error between binary outcomes and probabilities."""
    target = _as_vector(y_true, "y_true")
    probability = _as_vector(y_prob, "y_prob")
    if target.shape != probability.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only 0 and 1")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("y_prob must be between 0 and 1")
    return float(np.mean(np.square(probability - target)))


def reference_brier_score(y_true: object) -> float:
    """Return r * (1 - r), where r is the observed positive rate."""
    target = _as_vector(y_true, "y_true")
    if not np.isin(target, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only 0 and 1")
    rate = float(np.mean(target))
    return rate * (1.0 - rate)


def brier_skill_score(y_true: object, y_prob: object) -> float:
    """Compare a forecast with the constant observed-rate forecast."""
    reference = reference_brier_score(y_true)
    if reference == 0.0:
        raise ValueError("Brier skill score is undefined for a constant target")
    return 1.0 - brier_score(y_true, y_prob) / reference

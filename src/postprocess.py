"""Numerically stable probability post-processing."""

from __future__ import annotations

import numpy as np


def clip_probability(values: object, epsilon: float = 1e-6) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.clip(array, epsilon, 1.0 - epsilon)


def logit(values: object, epsilon: float = 1e-6) -> np.ndarray:
    probability = clip_probability(values, epsilon)
    return np.log(probability / (1.0 - probability))


def sigmoid(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_value = np.exp(array[~positive])
    output[~positive] = exp_value / (1.0 + exp_value)
    return output


def add_clipped_residual(
    probability: object,
    residual: object,
    scale: float = 1.0,
    clip: float | None = None,
) -> np.ndarray:
    delta = np.asarray(residual, dtype=float) * float(scale)
    if clip is not None:
        delta = np.clip(delta, -float(clip), float(clip))
    return clip_probability(np.asarray(probability, dtype=float) + delta)


def pivot_logit_sharpen(
    probability: object,
    pivot_logit: float,
    factor: float,
) -> np.ndarray:
    """Scale distance from a logit pivot and map back to probability."""
    logits = logit(probability)
    sharpened = float(pivot_logit) + float(factor) * (logits - float(pivot_logit))
    return clip_probability(sigmoid(sharpened))

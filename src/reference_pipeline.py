"""Weight-free reconstruction of the verified v251 inference order.

This module does not contain model weights or lookup artifacts. Callers supply
base-model probabilities and already deployment-scaled residual contributions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .postprocess import add_clipped_residual, clip_probability, pivot_logit_sharpen


OVERLAY_ORDER = (
    "tm_auto5",
    "structural_state",
    "forward_expert",
    "pitcher_state",
    "log5_pitcher_state",
    "v156_residual",
    "v224_offset",
)


def load_reference_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def weighted_probability_blend(
    predictions: Mapping[str, object],
    normalized_weights: Mapping[str, float],
) -> np.ndarray:
    """Blend named probability arrays using normalized public weights."""
    missing = set(normalized_weights).difference(predictions)
    if missing:
        raise KeyError(f"missing member predictions: {sorted(missing)}")

    total_weight = float(sum(normalized_weights.values()))
    if not np.isclose(total_weight, 1.0, atol=1e-8):
        raise ValueError(f"normalized weights must sum to 1, got {total_weight}")

    result: np.ndarray | None = None
    expected_shape: tuple[int, ...] | None = None
    for name, weight in normalized_weights.items():
        probability = clip_probability(predictions[name])
        if expected_shape is None:
            expected_shape = probability.shape
            result = np.zeros_like(probability, dtype=float)
        elif probability.shape != expected_shape:
            raise ValueError("all member predictions must have the same shape")
        result += float(weight) * probability

    if result is None:
        raise ValueError("at least one model prediction is required")
    return clip_probability(result)


def run_v251_reference(
    member_probabilities: Mapping[str, object],
    residual_contributions: Mapping[str, object],
    config: Mapping[str, object],
) -> np.ndarray:
    """Apply the documented v251 blend, residual order, and final sharpening."""
    ensemble = config["ensemble"]
    probability = weighted_probability_blend(
        member_probabilities,
        ensemble["normalized_weights"],
    )

    overlay_settings = config["overlays"]
    for name in OVERLAY_ORDER:
        settings = overlay_settings[name]
        if not settings.get("enabled", True):
            continue
        if name not in residual_contributions:
            raise KeyError(f"missing residual contribution: {name}")
        probability = add_clipped_residual(
            probability,
            residual_contributions[name],
            scale=float(settings.get("scale", 1.0)),
            clip=settings.get("clip"),
        )

    sharpening = config["final_logit_sharpen"]
    return pivot_logit_sharpen(
        probability,
        pivot_logit=float(sharpening["pivot_logit"]),
        factor=float(sharpening["factor"]),
    )

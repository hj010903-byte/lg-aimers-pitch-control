"""Representative, leakage-aware public feature utilities.

The private competition pipeline contained additional artifact-backed features.
This module keeps the reusable feature logic that can be explained without
shipping competition data or trained parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


def add_row_local_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add features available from the current pre-pitch row only."""
    required = {"balls_before", "strikes_before"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    result = frame.copy()
    balls = pd.to_numeric(result["balls_before"], errors="coerce")
    strikes = pd.to_numeric(result["strikes_before"], errors="coerce")
    result["count_total"] = balls + strikes
    result["count_pressure"] = balls - strikes
    result["two_strike"] = strikes.eq(2).astype("int8")
    result["three_ball"] = balls.eq(3).astype("int8")

    if "base_state" in result:
        state = result["base_state"].fillna("empty").astype(str)
        result["runners_on"] = state.ne("empty").astype("int8")
    return result


@dataclass(frozen=True)
class SmoothedRateLookup:
    keys: tuple[str, ...]
    table: pd.DataFrame
    global_rate: float
    rate_column: str = "historical_success_rate"


def fit_smoothed_rate_lookup(
    history: pd.DataFrame,
    keys: Sequence[str],
    target: str = "control_success",
    smoothing: float = 50.0,
) -> SmoothedRateLookup:
    """Fit a target-rate lookup from historical rows only."""
    keys = tuple(keys)
    missing = set(keys).union({target}).difference(history.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")

    global_rate = float(history[target].mean())
    grouped = history.groupby(list(keys), dropna=False)[target].agg(["sum", "count"]).reset_index()
    grouped["historical_success_rate"] = (
        grouped["sum"] + smoothing * global_rate
    ) / (grouped["count"] + smoothing)
    table = grouped.loc[:, [*keys, "historical_success_rate"]]
    return SmoothedRateLookup(keys=keys, table=table, global_rate=global_rate)


def apply_smoothed_rate_lookup(
    frame: pd.DataFrame,
    lookup: SmoothedRateLookup,
) -> pd.DataFrame:
    """Join a historical lookup and fall back to its training-period rate."""
    result = frame.merge(lookup.table, on=list(lookup.keys), how="left", validate="m:1")
    result[lookup.rate_column] = result[lookup.rate_column].fillna(lookup.global_rate)
    return result

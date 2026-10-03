"""Season-aware validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonSplit:
    train_seasons: tuple[int, ...]
    validation_season: int
    train_index: np.ndarray
    validation_index: np.ndarray


def expanding_season_splits(
    frame: pd.DataFrame,
    validation_seasons: Iterable[int] = (2022, 2023, 2024),
    season_column: str = "season",
) -> list[SeasonSplit]:
    """Build expanding splits using only seasons earlier than validation."""
    if season_column not in frame:
        raise KeyError(f"missing season column: {season_column}")

    seasons = pd.to_numeric(frame[season_column], errors="raise").astype(int)
    available = set(seasons.unique().tolist())
    splits: list[SeasonSplit] = []

    for validation_season in validation_seasons:
        validation_season = int(validation_season)
        train_seasons = tuple(sorted(season for season in available if season < validation_season))
        if not train_seasons or validation_season not in available:
            continue

        train_mask = seasons.isin(train_seasons).to_numpy()
        validation_mask = seasons.eq(validation_season).to_numpy()
        splits.append(
            SeasonSplit(
                train_seasons=train_seasons,
                validation_season=validation_season,
                train_index=np.flatnonzero(train_mask),
                validation_index=np.flatnonzero(validation_mask),
            )
        )

    return splits

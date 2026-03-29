"""Compute multi-year temporal features: previous year, weighted averages, trends.

For each target stat (OBP, SLG, HR, R, RBI, SB), this module creates:
  - ``prev_year_{stat}``: same player's stat from season Y-1
  - ``weighted_avg_{stat}``: recency-weighted average over Y-1, Y-2, Y-3
  - ``trend_{stat}``: Y-1 minus Y-2 (momentum signal)

All lookbacks are strictly backward — no future data is ever used.
Non-consecutive seasons (player missed a year) are handled by checking that
lag seasons match the expected year offset. Missing lags are excluded from
weighted averages with adjusted weights.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default temporal weights: Y-1, Y-2, Y-3 (most recent first)
DEFAULT_WEIGHTS: list[int] = [5, 3, 2]

# Stats to compute temporal features for (the 6 target stats)
TEMPORAL_STATS: list[str] = ["obp", "slg", "hr", "r", "rbi", "sb"]


def compute_temporal_features(
    df: pd.DataFrame,
    stats: list[str] | None = None,
    weights: list[int] | None = None,
    stat_weights: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """Compute temporal features for all target stats.

    Parameters
    ----------
    df:
        Merged dataset with ``mlbam_id``, ``season``, and stat columns.
        Must be the full multi-season dataset (not a single-season slice).
    stats:
        Stats to compute temporal features for. Defaults to
        ``TEMPORAL_STATS`` (the 6 target stats).
    weights:
        Default recency weights for Y-1, Y-2, Y-3. Defaults to ``[5, 3, 2]``.
    stat_weights:
        Optional per-stat weight overrides. Keys are stat names, values
        are weight lists. Stats not in this dict use *weights*.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with temporal feature columns added:
        ``prev_year_{stat}``, ``weighted_avg_{stat}``, ``trend_{stat}``
        for each stat.
    """
    if stats is None:
        stats = TEMPORAL_STATS
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if stat_weights is None:
        stat_weights = {}

    df = df.sort_values(["mlbam_id", "season"]).copy()
    grouped = df.groupby("mlbam_id")

    # Pre-compute lagged season columns for gap detection
    lag_seasons: dict[int, pd.Series] = {}
    for lag in range(1, len(weights) + 1):
        lag_seasons[lag] = grouped["season"].shift(lag)

    for stat in stats:
        if stat not in df.columns:
            logger.warning(
                "Stat '%s' not found in DataFrame — skipping temporal features",
                stat,
            )
            for suffix in ("prev_year_", "weighted_avg_", "trend_"):
                df[f"{suffix}{stat}"] = np.nan
            continue

        # Use per-stat weights if available, otherwise fall back to defaults
        w_for_stat = stat_weights.get(stat, weights)

        # Compute lag values (shift within player groups)
        lag_values: dict[int, pd.Series] = {}
        for lag in range(1, len(w_for_stat) + 1):
            if lag not in lag_seasons:
                lag_seasons[lag] = grouped["season"].shift(lag)
            raw_lag = grouped[stat].shift(lag)
            expected_season = df["season"] - lag
            # Null out lags that don't correspond to consecutive seasons
            is_valid = lag_seasons[lag] == expected_season
            lag_values[lag] = raw_lag.where(is_valid, other=np.nan)

        # prev_year_{stat} = lag 1 (strictly Y-1)
        df[f"prev_year_{stat}"] = lag_values[1]

        # weighted_avg_{stat} = sum(w_i * val_i) / sum(w_i) for available lags
        numerator = pd.Series(0.0, index=df.index)
        denominator = pd.Series(0.0, index=df.index)
        for i, (lag_idx, w) in enumerate(zip(range(1, len(w_for_stat) + 1), w_for_stat)):
            valid_mask = lag_values[lag_idx].notna()
            numerator = numerator + valid_mask.astype(float) * w * lag_values[lag_idx].fillna(0.0)
            denominator = denominator + valid_mask.astype(float) * w

        df[f"weighted_avg_{stat}"] = np.where(
            denominator > 0, numerator / denominator, np.nan,
        )

        # trend_{stat} = Y-1 minus Y-2
        df[f"trend_{stat}"] = lag_values[1] - lag_values.get(2, pd.Series(np.nan, index=df.index))

    return df

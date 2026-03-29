"""Select and validate Statcast quality features.

Statcast columns (avg_exit_velocity, ev_p95, max_exit_velocity,
avg_launch_angle, barrel_rate, hard_hit_rate, sweet_spot_rate) are already
present in the merged dataset from the data pipeline. This module ensures
they exist, fills missing values with league medians, and returns the
validated DataFrame.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

STATCAST_COLS: list[str] = [
    "bbe_count",
    "avg_exit_velocity",
    "ev_p95",
    "max_exit_velocity",
    "avg_launch_angle",
    "barrel_rate",
    "hard_hit_rate",
    "sweet_spot_rate",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
]

# Sensible defaults when an entire column is missing
_DEFAULTS: dict[str, float] = {
    "bbe_count": 120.0,
    "avg_exit_velocity": 88.0,
    "ev_p95": 104.0,
    "max_exit_velocity": 110.0,
    "avg_launch_angle": 12.0,
    "barrel_rate": 0.065,
    "hard_hit_rate": 0.35,
    "sweet_spot_rate": 0.33,
    "estimated_woba_using_speedangle": 0.320,
    "estimated_ba_using_speedangle": 0.250,
    "estimated_slg_using_speedangle": 0.410,
}


def compute_statcast_features(
    df: pd.DataFrame,
    add_missing_indicators: bool = False,
) -> pd.DataFrame:
    """Validate and impute Statcast quality features.

    Parameters
    ----------
    df:
        Merged dataset that may contain Statcast columns from the data
        pipeline merge step.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with all ``STATCAST_COLS`` guaranteed present.
        Missing values are filled with per-column league medians (or
        hard-coded defaults if the entire column is absent).
    """
    df = df.copy()

    for col in STATCAST_COLS:
        if col not in df.columns:
            logger.warning(
                "Statcast column '%s' missing from data — filling with default %.3f",
                col, _DEFAULTS[col],
            )
            df[col] = _DEFAULTS[col]
            if add_missing_indicators:
                df[f"has_{col}"] = 0
        else:
            if add_missing_indicators:
                df[f"has_{col}"] = df[col].notna().astype(int)
            median_val = df[col].median()
            fill_val = median_val if pd.notna(median_val) else _DEFAULTS[col]
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                logger.info(
                    "Imputing %d missing values in '%s' with %.3f",
                    n_missing, col, fill_val,
                )
                df[col] = df[col].fillna(fill_val)

    return df

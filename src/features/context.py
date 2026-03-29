"""Validate and prepare context features.

Context features (age, age_squared, park factors, team stats) are already
present in the merged dataset from the data pipeline. This module ensures
they exist and handles any remaining missing values.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default values for missing context data
_PARK_DEFAULTS: dict[str, float] = {
    "park_factor_runs": 1.0,
    "park_factor_hr": 1.0,
}

_TEAM_DEFAULTS: dict[str, float] = {
    "team_runs_per_game": 4.5,
    "team_ops": 0.720,
    "team_sb": 80.0,
}


def compute_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and fill context features (age, park factors, team stats).

    Parameters
    ----------
    df:
        Merged dataset that should contain ``age``, park factor, and team
        stat columns from the data pipeline.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with all context columns guaranteed present and
        non-null.
    """
    df = df.copy()

    # Age features
    if "age" not in df.columns:
        logger.warning("'age' column missing — filling with league average 28")
        df["age"] = 28.0
    else:
        df["age"] = df["age"].fillna(df["age"].median())

    if "age_squared" not in df.columns:
        df["age_squared"] = df["age"] ** 2
    else:
        # Recompute to ensure consistency
        df["age_squared"] = df["age"] ** 2

    # Park factors
    for col, default in _PARK_DEFAULTS.items():
        if col not in df.columns:
            logger.warning("'%s' missing — filling with neutral %.1f", col, default)
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # Team stats
    for col, default in _TEAM_DEFAULTS.items():
        if col not in df.columns:
            logger.warning("'%s' missing — filling with default %.1f", col, default)
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # Team SB per game (fallback if not pre-computed in fetch_context).
    # fetch_context already normalizes by actual games played (including 2020's 60).
    # This fallback is rarely exercised; uses 162 as the standard divisor since
    # any shortened-season handling belongs in the data pipeline, not features.
    if "team_sb_per_game" not in df.columns:
        if "team_sb" in df.columns:
            df["team_sb_per_game"] = df["team_sb"] / 162.0
        else:
            df["team_sb_per_game"] = 0.494  # league average fallback
    # Fill NaN values from failed team joins (column exists but null)
    df["team_sb_per_game"] = df["team_sb_per_game"].fillna(0.494)

    # SB rule-era indicator (2023 rule change: larger bases, limited pickoffs)
    if "season" in df.columns:
        df["sb_rule_era"] = (df["season"] >= 2023).astype(float)
    else:
        df["sb_rule_era"] = 0.0

    # Era-conditioned speed: captures that same sprint_speed produces
    # far more SBs post-2023 than pre-2023
    sprint = df["sprint_speed"] if "sprint_speed" in df.columns else 27.0
    df["sb_era_x_speed"] = df["sb_rule_era"] * sprint

    # Speed-aging interaction: captures accelerating speed decline after peak
    df["speed_age_interaction"] = sprint * (df["age"] - 27.0)

    # Era-conditioned attempt rate: captures that the 2023 rule change
    # affects aggressive baserunners (high attempt rate) more
    if "sb_attempt_rate" in df.columns:
        df["sb_era_x_attempt_rate"] = df["sb_rule_era"] * df["sb_attempt_rate"]
    else:
        df["sb_era_x_attempt_rate"] = 0.0

    # Stat-specific aging curves (piecewise-linear from FanGraphs research)
    age = df["age"]

    # Speed: peaks ~23, steep decline after 27
    df["age_delta_speed"] = np.select(
        [age <= 23, age <= 27],
        [0.0, -0.005 * (age - 23)],
        default=-0.02 - 0.015 * (age - 27),
    )

    # Power: peaks ~27, plateau to 30, then decline
    df["age_delta_power"] = np.select(
        [age <= 27, age <= 30],
        [0.005 * (age - 22), 0.025],
        default=0.025 - 0.01 * (age - 30),
    )

    # Patience (BB%, OBP): peaks ~30, slow decline after 33
    df["age_delta_patience"] = np.select(
        [age <= 30, age <= 33],
        [0.003 * (age - 24), 0.018],
        default=0.018 - 0.005 * (age - 33),
    )

    return df

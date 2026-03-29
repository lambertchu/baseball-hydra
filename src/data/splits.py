"""Chronological train/val/test split logic.

Splits are strictly by season year to prevent temporal leakage.
Features from year Y predict targets from year Y+1, so:
  - Train: features from seasons [start_year, train_end]
  - Val:   features from season val_year
  - Test:  features from season test_year

The target for a train row from season 2022 is the player's 2023 stats.
Rows without valid targets (NaN) are dropped.

Usage
-----
    from src.data.splits import split_data, SplitConfig

    config = SplitConfig(train_end=2022, val_year=2023, test_year=2024)
    train, val, test = split_data(df, config)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for chronological data splits.

    Parameters
    ----------
    train_end:
        Last season included in training features. Training features span
        all seasons up to and including this year.
    val_year:
        Season year for validation features.
    test_year:
        Season year for test features.
    """

    train_end: int
    val_year: int
    test_year: int

    def __post_init__(self) -> None:
        if not (self.train_end < self.val_year < self.test_year):
            raise ValueError(
                f"Split years must be strictly increasing: "
                f"train_end={self.train_end} < val_year={self.val_year} "
                f"< test_year={self.test_year}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> SplitConfig:
        """Create from a config dict (e.g. from YAML).

        Supports two formats:

        1. **Target-year** (preferred): specify ``test_target_year`` and
           the feature years are derived automatically::

               {"test_target_year": 2025}
               # → train_end=2022, val_year=2023, test_year=2024

        2. **Explicit feature years** (backward compatible)::

               {"train_end": 2022, "val_year": 2023, "test_year": 2024}
        """
        if "test_target_year" in d:
            target_year = d["test_target_year"]
            return cls(
                train_end=target_year - 3,
                val_year=target_year - 2,
                test_year=target_year - 1,
            )
        return cls(
            train_end=d["train_end"],
            val_year=d["val_year"],
            test_year=d["test_year"],
        )


def split_data(
    df: pd.DataFrame,
    config: SplitConfig,
    target_cols: list[str] | None = None,
    drop_na_targets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a merged dataset into train, validation, and test sets.

    Parameters
    ----------
    df:
        Merged dataset with ``season`` column and target columns.
    config:
        Split boundaries.
    target_cols:
        Names of target columns (e.g. ["target_obp", "target_slg", ...]).
        If provided and drop_na_targets is True, rows with any NaN target
        are dropped.
    drop_na_targets:
        If True, drop rows where any target column is NaN.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train, val, test) DataFrames.
    """
    train = df[df["season"] <= config.train_end].copy()
    val = df[df["season"] == config.val_year].copy()
    test = df[df["season"] == config.test_year].copy()

    if drop_na_targets and target_cols:
        available = [c for c in target_cols if c in df.columns]
        if available:
            train = train.dropna(subset=available)
            val = val.dropna(subset=available)
            test = test.dropna(subset=available)

    return train, val, test


def get_production_data(
    df: pd.DataFrame,
    end_year: int,
    target_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data for production (2026) predictions.

    For production, we retrain on all available data with valid targets
    and use the most recent season's features for prediction.

    Parameters
    ----------
    df:
        Full merged dataset.
    end_year:
        Most recent season with features (e.g. 2025).
    target_cols:
        Names of target columns.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (retrain_data, predict_data)
        - retrain_data: all rows with valid targets (for retraining)
        - predict_data: rows from end_year (features only, targets are unknown)
    """
    # Retrain on all rows that have valid targets
    retrain = df[df["season"] < end_year].copy()
    if target_cols:
        available = [c for c in target_cols if c in retrain.columns]
        if available:
            retrain = retrain.dropna(subset=available)

    # Predict for the latest season (targets are NaN — that's what we predict)
    predict = df[df["season"] == end_year].copy()

    return retrain, predict

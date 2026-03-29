"""Rolling-origin backtest utilities.

Defines the canonical five-fold rolling-origin schedule used for MTL:

- Fold A: train<=2019, val=2020, test=2021
- Fold B: train<=2020, val=2021, test=2022
- Fold C: train<=2021, val=2022, test=2023
- Fold D: train<=2022, val=2023, test=2024
- Fold E: train<=2023, val=2024, test=2025
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.splits import SplitConfig, split_data


@dataclass(frozen=True)
class BacktestFold:
    """One rolling-origin backtest fold definition."""

    name: str
    train_end: int
    val_year: int
    test_year: int

    def to_split_config(self) -> SplitConfig:
        """Convert to SplitConfig."""
        return SplitConfig(
            train_end=self.train_end,
            val_year=self.val_year,
            test_year=self.test_year,
        )


DEFAULT_BACKTEST_FOLDS: tuple[BacktestFold, ...] = (
    BacktestFold(name="A", train_end=2019, val_year=2020, test_year=2021),
    BacktestFold(name="B", train_end=2020, val_year=2021, test_year=2022),
    BacktestFold(name="C", train_end=2021, val_year=2022, test_year=2023),
    BacktestFold(name="D", train_end=2022, val_year=2023, test_year=2024),
    BacktestFold(name="E", train_end=2023, val_year=2024, test_year=2025),
)


def load_backtest_folds(config: dict | None = None) -> list[BacktestFold]:
    """Load backtest folds from model config, falling back to defaults."""
    if not config:
        return list(DEFAULT_BACKTEST_FOLDS)

    fold_rows = config.get("backtest", {}).get("folds", [])
    if not fold_rows:
        return list(DEFAULT_BACKTEST_FOLDS)

    folds: list[BacktestFold] = []
    for row in fold_rows:
        folds.append(
            BacktestFold(
                name=str(row["name"]),
                train_end=int(row["train_end"]),
                val_year=int(row["val_year"]),
                test_year=int(row["test_year"]),
            )
        )
    return folds


def split_for_fold(
    df: pd.DataFrame,
    fold: BacktestFold,
    target_cols: list[str] | None = None,
    drop_na_targets: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/val/test splits for one fold."""
    return split_data(
        df=df,
        config=fold.to_split_config(),
        target_cols=target_cols,
        drop_na_targets=drop_na_targets,
    )


def iter_backtest_splits(
    df: pd.DataFrame,
    folds: list[BacktestFold],
    target_cols: list[str] | None = None,
    drop_na_targets: bool = True,
):
    """Yield ``(fold, train_df, val_df, test_df)`` for each configured fold."""
    for fold in folds:
        train_df, val_df, test_df = split_for_fold(
            df=df,
            fold=fold,
            target_cols=target_cols,
            drop_na_targets=drop_na_targets,
        )
        yield fold, train_df, val_df, test_df

"""End-to-end feature pipeline: merged data → feature matrix + target vector.

Orchestrates all feature computation steps (batting, non-contact, statcast,
contact quality, context, temporal) and extracts the final (X, y) pair ready
for modeling.

Usage
-----
    from src.features.pipeline import build_features, extract_xy

    df = build_features(merged_df, config)
    X_train, y_train = extract_xy(train_df, config)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.batting import compute_batting_features
from src.features.context import compute_context_features
from src.features.non_contact import compute_non_contact_features
from src.features.registry import RATE_DECOMP_STATS, TARGET_COLUMNS, get_feature_names
from src.features.statcast import compute_statcast_features
from src.features.temporal import compute_temporal_features

logger = logging.getLogger(__name__)

_EXPECTED_CONTACT_COLS = [
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
]

_BAT_TRACKING_EXTENDED_COLS = [
    "squared_up_rate",
    "blast_rate",
    "fast_swing_rate",
    "bat_tracking_swings",
    "bat_tracking_bbe",
    "bat_tracking_blasts",
    "bat_tracking_squared_up",
    "bat_tracking_fast_swings",
    "has_squared_up_rate",
    "has_blast_rate",
    "has_fast_swing_rate",
    "has_bat_tracking_swings",
    "has_bat_tracking_bbe",
    "has_bat_tracking_blasts",
    "has_bat_tracking_squared_up",
    "has_bat_tracking_fast_swings",
]


def build_features(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline on merged data.

    Parameters
    ----------
    df:
        Merged dataset from ``src.data.merge.run_merge``. Must contain
        raw batting, Statcast, speed, context, and target columns.
    config:
        Data pipeline config dict. If None, defaults are used.

    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns computed and added.
        Original columns are preserved alongside new derived features.
    """
    if config is None:
        config = {}

    feature_groups = config.get("feature_groups", {})
    feature_options = config.get("feature_options", {})
    temporal_weights = config.get("temporal_weights", [5, 3, 2])
    temporal_stat_weights: dict[str, list[int]] = config.get("temporal_stat_weights", {})
    add_missing_indicators = bool(feature_options.get("missing_indicators", False))
    include_expected_contact = bool(feature_options.get("expected_contact_stats", True))
    include_bat_tracking_extended = bool(feature_options.get("bat_tracking_extended", True))

    logger.info("Building features for %d rows …", len(df))

    # 1. Derived batting features (bb_rate, k_rate, iso, sb_rate, hbp_rate, contact_rate)
    if feature_groups.get("batting", True):
        logger.info("  Computing batting features …")
        df = compute_batting_features(df)

    # 1b. Non-contact features (stabilisation-regressed K%, BB%, HBP%)
    if feature_groups.get("non_contact", True):
        logger.info("  Computing non-contact features …")
        df = compute_non_contact_features(df)

    # 2. Statcast quality features (validate + impute)
    if feature_groups.get("statcast", True):
        logger.info("  Validating Statcast features …")
        df = compute_statcast_features(
            df,
            add_missing_indicators=add_missing_indicators,
        )
        if not include_expected_contact:
            drop_cols = [c for c in _EXPECTED_CONTACT_COLS if c in df.columns]
            drop_cols += [f"has_{c}" for c in _EXPECTED_CONTACT_COLS if f"has_{c}" in df.columns]
            if drop_cols:
                df = df.drop(columns=drop_cols)

    if not include_bat_tracking_extended:
        drop_cols = [c for c in _BAT_TRACKING_EXTENDED_COLS if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    # 3. Context features (age, park factors, team stats)
    if feature_groups.get("age", True) or feature_groups.get("park_factors", True) or feature_groups.get("team_stats", True):
        logger.info("  Validating context features …")
        df = compute_context_features(df)

    # 4. Temporal features (prev_year, weighted_avg, trend) — single consolidated call
    if feature_groups.get("temporal", True):
        from src.features.registry import TARGET_STATS

        all_temporal_stats: list[str] = list(TARGET_STATS)
        if feature_groups.get("batting", True):
            all_temporal_stats.extend(s for s in RATE_DECOMP_STATS if s in df.columns)
        if feature_groups.get("statcast", True):
            all_temporal_stats.extend(c for c in _EXPECTED_CONTACT_COLS if c in df.columns)
        # PA temporal features (prev_year_pa) needed for Marcel PA projection
        if "pa" in df.columns and "pa" not in all_temporal_stats:
            all_temporal_stats.append("pa")

        # Deduplicate while preserving order
        unique_stats = list(dict.fromkeys(all_temporal_stats))

        if unique_stats:
            logger.info("  Computing temporal features for %d stats …", len(unique_stats))
            df = compute_temporal_features(df, stats=unique_stats, weights=temporal_weights, stat_weights=temporal_stat_weights)

    logger.info("  Feature engineering complete — %d columns total", len(df.columns))
    return df


def extract_xy(
    df: pd.DataFrame,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract feature matrix X and target matrix y from a feature-engineered DataFrame.

    Parameters
    ----------
    df:
        Feature-engineered DataFrame (output of ``build_features``).
    config:
        Data pipeline config dict with ``feature_groups`` section. If None,
        all feature groups are enabled.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(X, y)`` where X has shape (n_samples, n_features) and y has
        shape (n_samples, n_targets). Column names are preserved.
    """
    if config is None:
        config = {}

    enabled_groups = config.get("feature_groups", None)
    exclude_features: list[str] = config.get("exclude_features", [])
    feature_names = get_feature_names(enabled_groups)

    # Select only features that actually exist in the DataFrame, minus exclusions
    available_features = [f for f in feature_names if f in df.columns and f not in exclude_features]
    missing = set(feature_names) - set(available_features) - set(exclude_features)
    if missing:
        logger.warning("Features not found in DataFrame (will be skipped): %s", sorted(missing))

    X = df[available_features].copy()
    # Convert nullable pandas dtypes (Float64, Int64) to standard float64
    # so that pd.NA values become np.nan (numpy cannot convert pd.NA).
    X = X.astype(np.float64)

    y = df[[c for c in TARGET_COLUMNS if c in df.columns]].copy()
    y = y.astype(np.float64)

    logger.info(
        "Extracted X: (%d, %d)  y: (%d, %d)",
        X.shape[0], X.shape[1], y.shape[0], y.shape[1],
    )
    return X, y


def run_feature_pipeline(
    merged_path: str | Path,
    config_path: str | Path = "configs/data.yaml",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load merged data, build features, and optionally save.

    Convenience function for CLI usage.

    Parameters
    ----------
    merged_path:
        Path to the merged Parquet file.
    config_path:
        Path to the data config YAML.
    output_path:
        If provided, save the feature-engineered DataFrame to this path.

    Returns
    -------
    pd.DataFrame
        Feature-engineered DataFrame.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info("Loading merged data from %s …", merged_path)
    df = pd.read_parquet(merged_path)

    df = build_features(df, config)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        size_mb = out.stat().st_size / 1_048_576
        logger.info("Saved feature matrix → %s  (%.1f MB)", out, size_mb)

    return df

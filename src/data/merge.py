"""Merge all data sources into a single modeling dataset.

Joins FanGraphs batting stats, Statcast aggregated metrics, sprint speed,
bat speed, park factors, and team stats on (player, season). Aligns targets
so that features from year Y have targets from year Y+1.

Usage
-----
    python -m src.data.merge --config configs/data.yaml

Output
------
    data/merged_batter_data.parquet

Raw per-year parquet files are read from ``data/raw/``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.registry import RATE_STATS

logger = logging.getLogger(__name__)

# Target stat columns in the batting data
_TARGET_COLS = ["obp", "slg", "hr", "r", "rbi", "sb"]

# FanGraphs → MLBAM ID mapping cache to avoid repeated lookups
_ID_MAP_CACHE: pd.DataFrame | None = None


def load_yaml_config(config_path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_parquet_years(
    pattern: str,
    data_dir: Path,
    years: list[int],
) -> pd.DataFrame:
    """Load and concatenate Parquet files matching a naming pattern.

    Parameters
    ----------
    pattern:
        Filename pattern with ``{year}`` placeholder, e.g. ``"batting_{year}.parquet"``.
    data_dir:
        Directory containing the Parquet files.
    years:
        Season years to load.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame from all available years.
    """
    frames: list[pd.DataFrame] = []
    for year in sorted(years):
        path = data_dir / pattern.format(year=year)
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            logger.warning("Missing data file: %s", path)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_id_map(batting_df: pd.DataFrame) -> pd.DataFrame:
    """Build a FanGraphs IDfg → MLBAM ID mapping table.

    Uses pybaseball.playerid_reverse_lookup to map all unique FanGraphs IDs
    in the batting data to MLBAM IDs used by Statcast.

    Parameters
    ----------
    batting_df:
        FanGraphs batting data with an ``idfg`` column.

    Returns
    -------
    pd.DataFrame
        Mapping table with columns ``idfg`` and ``mlbam_id``.
    """
    global _ID_MAP_CACHE
    if _ID_MAP_CACHE is not None:
        return _ID_MAP_CACHE

    try:
        import pybaseball as pb
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for ID mapping. Install with: uv sync"
        ) from exc

    fg_ids = batting_df["idfg"].dropna().astype(int).unique().tolist()
    logger.info("Building ID map for %d FanGraphs IDs …", len(fg_ids))

    id_map = pb.playerid_reverse_lookup(fg_ids, key_type="fangraphs")

    result = pd.DataFrame({
        "idfg": pd.to_numeric(id_map["key_fangraphs"], errors="coerce").astype("Int64"),
        "mlbam_id": pd.to_numeric(id_map["key_mlbam"], errors="coerce").astype("Int64"),
    })
    result = result.dropna(subset=["idfg", "mlbam_id"])
    result["idfg"] = result["idfg"].astype(int)
    result["mlbam_id"] = result["mlbam_id"].astype(int)
    result = result.drop_duplicates(subset=["idfg"], keep="first")

    logger.info("  Mapped %d / %d IDs successfully", len(result), len(fg_ids))
    _ID_MAP_CACHE = result
    return result


def merge_batting_with_statcast(
    batting: pd.DataFrame,
    statcast: pd.DataFrame,
    id_map: pd.DataFrame,
) -> pd.DataFrame:
    """Merge FanGraphs batting stats with Statcast aggregated metrics.

    Parameters
    ----------
    batting:
        FanGraphs batting data with ``idfg`` and ``season`` columns.
    statcast:
        Statcast aggregated data with ``mlbam_id`` and ``season`` columns.
    id_map:
        FanGraphs → MLBAM ID mapping with ``idfg`` and ``mlbam_id`` columns.

    Returns
    -------
    pd.DataFrame
        Merged data on (mlbam_id, season).
    """
    # Add mlbam_id to batting data
    batting_with_id = batting.merge(id_map, on="idfg", how="inner")

    if statcast.empty:
        return batting_with_id

    merged = batting_with_id.merge(
        statcast,
        on=["mlbam_id", "season"],
        how="left",
    )
    return merged


def merge_speed_data(
    df: pd.DataFrame,
    sprint_df: pd.DataFrame,
    bat_speed_df: pd.DataFrame,
    bat_speed_impute: str = "league_median",
) -> pd.DataFrame:
    """Merge sprint speed and bat speed data into the main dataset.

    Parameters
    ----------
    df:
        Main merged dataset with ``mlbam_id`` and ``season`` columns.
    sprint_df:
        Sprint speed data.
    bat_speed_df:
        Bat speed data (may be empty for years < 2024).
    bat_speed_impute:
        How to impute missing bat speed: "league_median" or "zero".

    Returns
    -------
    pd.DataFrame
        Dataset with speed columns added.
    """
    if not sprint_df.empty:
        df = df.merge(sprint_df, on=["mlbam_id", "season"], how="left")

    if not bat_speed_df.empty:
        df = df.merge(bat_speed_df, on=["mlbam_id", "season"], how="left")

    # Impute bat-tracking columns and expose missingness indicators
    bat_speed_numeric = [
        "avg_bat_speed",
        "avg_swing_speed",
        "squared_up_rate",
        "blast_rate",
        "fast_swing_rate",
        "bat_tracking_swings",
        "bat_tracking_bbe",
        "bat_tracking_blasts",
        "bat_tracking_squared_up",
        "bat_tracking_fast_swings",
    ]
    count_like = {
        "bat_tracking_swings",
        "bat_tracking_bbe",
        "bat_tracking_blasts",
        "bat_tracking_squared_up",
        "bat_tracking_fast_swings",
    }

    for col in bat_speed_numeric:
        if col in df.columns:
            # Indicator is captured before imputation.
            df[f"has_{col}"] = df[col].notna().astype(int)
            if col in count_like:
                df[col] = df[col].fillna(0.0)
            elif bat_speed_impute == "league_median":
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0.0)
            else:
                df[col] = df[col].fillna(0.0)
        else:
            default_val = 0.0
            df[col] = default_val
            df[f"has_{col}"] = 0

    # Impute missing sprint speed with position-group median
    if "sprint_speed" in df.columns:
        if "has_sprint_speed" not in df.columns:
            # Marker before fill for optional missing-indicator features.
            df["has_sprint_speed"] = df["sprint_speed"].notna().astype(int)
        median_speed = df["sprint_speed"].median()
        df["sprint_speed"] = df["sprint_speed"].fillna(
            median_speed if pd.notna(median_speed) else 27.0  # ~league avg
        )
    else:
        df["sprint_speed"] = 27.0
        df["has_sprint_speed"] = 0

    return df


def merge_context_data(
    df: pd.DataFrame,
    park_factors_df: pd.DataFrame,
    team_batting_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge park factors and team batting stats into the main dataset.

    Parameters
    ----------
    df:
        Main merged dataset with ``team`` and ``season`` columns.
    park_factors_df:
        Park factors data.
    team_batting_df:
        Team batting stats.

    Returns
    -------
    pd.DataFrame
        Dataset with context columns added.
    """
    if not park_factors_df.empty:
        df = df.merge(park_factors_df, on=["team", "season"], how="left")
        # Fill missing park factors with neutral (1.0)
        for col in ["park_factor_runs", "park_factor_hr"]:
            if col in df.columns:
                df[col] = df[col].fillna(1.0)

    if not team_batting_df.empty:
        df = df.merge(team_batting_df, on=["team", "season"], how="left")
        # Fill missing team stats with league averages
        defaults = {"team_runs_per_game": 4.5, "team_ops": 0.720, "team_sb": 80}
        for col, default in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

    return df




def align_targets(
    df: pd.DataFrame,
    targets: list[str],
    rate_targets: bool = False,
) -> pd.DataFrame:
    """Create target columns by aligning next-season stats.

    For each (player, season Y), the target is the player's stats in season Y+1.
    Rows where Y+1 data is unavailable (e.g. the most recent season for active
    players, or players who didn't play the next year) will have NaN targets.

    Parameters
    ----------
    df:
        Dataset with ``mlbam_id``, ``season``, and target stat columns.
    targets:
        List of target column names (e.g. ["obp", "slg", "hr", "r", "rbi", "sb"]).
    rate_targets:
        If True, convert count targets (HR, R, RBI, SB) to per-PA rates by
        dividing by next-season PA. Rate targets (OBP, SLG) and PA itself
        are left unchanged. This aligns with how professional projection
        systems decompose predictions into rate × playing time.

    Returns
    -------
    pd.DataFrame
        Dataset with ``target_{stat}`` columns added. Original stat columns
        remain as features.
    """
    df = df.sort_values(["mlbam_id", "season"]).copy()

    # Vectorized approach: within each player group, shift target columns
    # up by one row so that row i gets the values from row i+1.
    grouped = df.groupby("mlbam_id")
    for target in targets:
        df[f"target_{target}"] = grouped[target].shift(-1)

    # shift(-1) only works for consecutive rows within a group. If a player
    # skips a season (e.g., has 2021 and 2023 but not 2022), the 2021 row
    # would incorrectly get 2023 values. Null out rows where the next row's
    # season is not exactly current_season + 1.
    next_season = grouped["season"].shift(-1)
    non_consecutive = next_season != (df["season"] + 1)
    target_cols = [f"target_{t}" for t in targets]
    df.loc[non_consecutive, target_cols] = np.nan

    # Convert count targets to per-PA rates
    if rate_targets:
        # Ensure target_pa exists (may already be in targets list)
        if "target_pa" not in df.columns:
            df["target_pa"] = grouped["pa"].shift(-1)
            df.loc[non_consecutive, "target_pa"] = np.nan

        next_pa = df["target_pa"]
        count_targets = [
            t for t in targets
            if t.lower() not in RATE_STATS and t != "pa"
        ]
        for target in count_targets:
            col = f"target_{target}"
            df[col] = np.where(
                next_pa.notna() & (next_pa > 0),
                df[col] / next_pa,
                np.nan,
            )
        n_converted = len(count_targets)
        logger.info(
            "  Converted %d count targets to per-PA rates: %s",
            n_converted,
            [t.upper() for t in count_targets],
        )

    return df


def run_merge(config: dict) -> pd.DataFrame:
    """Execute the full merge pipeline from config.

    Parameters
    ----------
    config:
        Data pipeline configuration dict (from configs/data.yaml).

    Returns
    -------
    pd.DataFrame
        Fully merged dataset with features and aligned targets.
    """
    data_dir = Path(config.get("raw_data_dir", "data/raw"))
    start_year = config["start_year"]
    end_year = config["end_year"]
    min_pa = config.get("min_pa_train", 200)
    targets = config.get("targets", _TARGET_COLS)
    feature_opts = config.get("feature_options", {})
    bat_speed_impute = config.get("bat_speed_impute_method", "league_median")

    years = list(range(start_year, end_year + 1))
    logger.info("Merging data for seasons %d-%d …", start_year, end_year)

    # 1. Load all cached data sources
    logger.info("Loading batting stats …")
    batting = _load_parquet_years("batting_{year}.parquet", data_dir, years)
    if batting.empty:
        raise FileNotFoundError(
            f"No batting data found in {data_dir}. "
            "Fetch it first with: python -m src.data.fetch_batting --seasons 2016-2025"
        )

    # Filter by minimum PA
    batting = batting[batting["pa"] >= min_pa].copy()
    logger.info("  %d player-seasons after PA >= %d filter", len(batting), min_pa)

    logger.info("Loading Statcast aggregated metrics …")
    statcast = _load_parquet_years("statcast_agg_{year}.parquet", data_dir, years)

    logger.info("Loading sprint speed …")
    sprint = _load_parquet_years("sprint_speed_{year}.parquet", data_dir, years)

    logger.info("Loading bat speed …")
    bat_speed = _load_parquet_years("bat_speed_{year}.parquet", data_dir, years)

    logger.info("Loading park factors …")
    park_factors = _load_parquet_years("park_factors_{year}.parquet", data_dir, years)

    logger.info("Loading team batting …")
    team_batting = _load_parquet_years("team_batting_{year}.parquet", data_dir, years)

    # 2. Build FanGraphs → MLBAM ID map
    id_map = build_id_map(batting)

    # 3. Merge batting + Statcast
    logger.info("Merging batting + Statcast …")
    merged = merge_batting_with_statcast(batting, statcast, id_map)
    logger.info("  %d rows after batting + Statcast merge", len(merged))

    # 4. Merge speed data
    logger.info("Merging speed data …")
    merged = merge_speed_data(merged, sprint, bat_speed, bat_speed_impute)

    # 5. Merge context data (park factors + team stats)
    logger.info("Merging context data …")
    merged = merge_context_data(merged, park_factors, team_batting)

    # 6. Align targets (Y → Y+1)
    logger.info("Aligning targets (Y → Y+1) …")
    rate_targets = config.get("rate_targets", False)
    target_cols = [t.lower() for t in targets]
    # Always create target_pa when rate_targets or pa_target is enabled
    if (feature_opts.get("pa_target", False) or rate_targets) and "pa" not in target_cols:
        target_cols.append("pa")
    merged = align_targets(merged, target_cols, rate_targets=rate_targets)

    # 7. Add age-squared
    if "age" in merged.columns:
        merged["age_squared"] = merged["age"] ** 2

    # Report stats
    target_col_names = [f"target_{t}" for t in target_cols]
    has_targets = merged[target_col_names].notna().all(axis=1)
    logger.info(
        "Merge complete: %d total rows, %d with all targets",
        len(merged), has_targets.sum(),
    )

    return merged


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Merge all data sources into a single modeling dataset.",
    )
    parser.add_argument(
        "--config", default="configs/data.yaml", metavar="PATH",
        help="Path to data config YAML (default: configs/data.yaml)",
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    merged = run_merge(config)

    out_path = Path(config.get("output", {}).get("merged_dataset", "data/merged_batter_data.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)

    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("Saved merged dataset → %s  (%.1f MB, %d rows)", out_path, size_mb, len(merged))


if __name__ == "__main__":
    main()

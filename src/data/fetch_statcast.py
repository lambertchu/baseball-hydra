"""Fetch and cache aggregated Statcast batted-ball metrics per batter-season.

This module downloads raw Statcast pitch-level data for each season, filters
to batted-ball events, and aggregates per-batter quality metrics: average exit
velocity, max exit velocity, average launch angle, barrel rate, hard hit rate,
and sweet spot rate.

Usage
-----
    python -m src.data.fetch_statcast --seasons 2016-2025 --out-dir data/raw

Output
------
    data/raw/statcast_agg_{year}.parquet  (zstd-compressed)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Season date ranges for Statcast queries (regular season + postseason).
_SEASON_DATES: dict[int, tuple[str, str]] = {
    2016: ("2016-04-03", "2016-11-03"),
    2017: ("2017-04-02", "2017-11-02"),
    2018: ("2018-03-29", "2018-10-29"),
    2019: ("2019-03-28", "2019-10-31"),
    2020: ("2020-07-23", "2020-10-28"),
    2021: ("2021-04-01", "2021-11-03"),
    2022: ("2022-04-07", "2022-11-06"),
    2023: ("2023-03-30", "2023-11-02"),
    2024: ("2024-03-20", "2024-10-31"),
    2025: ("2025-03-27", "2025-10-31"),
}


def _aggregate_batter_statcast(df: pd.DataFrame, min_bbe: int = 50) -> pd.DataFrame:
    """Aggregate raw Statcast pitch data into per-batter quality metrics.

    Parameters
    ----------
    df:
        Raw Statcast pitch-level DataFrame (from pybaseball.statcast).
    min_bbe:
        Minimum batted-ball events for inclusion.

    Returns
    -------
    pd.DataFrame
        One row per batter with aggregated Statcast quality metrics.
    """
    # Filter to batted-ball events only (has launch_speed and launch_angle)
    bbe = df[
        df["launch_speed"].notna()
        & df["launch_angle"].notna()
        & df["bb_type"].notna()
        & (df["game_type"] == "R")
    ].copy()

    if bbe.empty:
        return pd.DataFrame()

    grouped = bbe.groupby("batter")

    agg = pd.DataFrame({
        "bbe_count": grouped.size(),
        "avg_exit_velocity": grouped["launch_speed"].mean(),
        "ev_p95": grouped["launch_speed"].quantile(0.95),
        "max_exit_velocity": grouped["launch_speed"].max(),
        "avg_launch_angle": grouped["launch_angle"].mean(),
    })

    # Expected-contact quality metrics (when present in Statcast payload)
    expected_cols = [
        "estimated_woba_using_speedangle",
        "estimated_ba_using_speedangle",
        "estimated_slg_using_speedangle",
    ]
    for col in expected_cols:
        if col in bbe.columns:
            agg[col] = grouped[col].mean()
        else:
            agg[col] = np.nan

    # Barrel rate: launch_speed_angle == 6 is the barrel classification
    if "launch_speed_angle" in bbe.columns:
        barrel_counts = bbe[bbe["launch_speed_angle"] == 6].groupby("batter").size()
        agg["barrel_rate"] = barrel_counts / agg["bbe_count"]
        agg["barrel_rate"] = agg["barrel_rate"].fillna(0.0)
    else:
        agg["barrel_rate"] = np.nan

    # Hard hit rate: exit velocity >= 95 mph
    hard_hit_counts = bbe[bbe["launch_speed"] >= 95.0].groupby("batter").size()
    agg["hard_hit_rate"] = hard_hit_counts / agg["bbe_count"]
    agg["hard_hit_rate"] = agg["hard_hit_rate"].fillna(0.0)

    # Sweet spot rate: launch angle 8-32 degrees
    sweet_spot_counts = (
        bbe[(bbe["launch_angle"] >= 8) & (bbe["launch_angle"] <= 32)]
        .groupby("batter")
        .size()
    )
    agg["sweet_spot_rate"] = sweet_spot_counts / agg["bbe_count"]
    agg["sweet_spot_rate"] = agg["sweet_spot_rate"].fillna(0.0)

    # Filter by minimum BBE
    agg = agg[agg["bbe_count"] >= min_bbe].copy()

    agg = agg.reset_index()
    agg = agg.rename(columns={"batter": "mlbam_id"})

    return agg


# Columns to keep when saving raw BBE data as a side-effect of API fetches.
# Defined locally (not imported from fetch_raw_statcast) to avoid circular imports.
_RAW_KEEP_COLUMNS = [
    "batter",
    "launch_speed",
    "launch_angle",
    "bb_type",
    "events",
    "hc_x",
    "hc_y",
    "game_type",
    "woba_value",
    "estimated_woba_using_speedangle",
    "launch_speed_angle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
]


def fetch_statcast(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 5.0,
    min_bbe: int = 50,
    from_api: bool = False,
) -> Path:
    """Aggregate per-batter Statcast metrics for one season and save.

    By default, reads from a cached ``statcast_raw_{year}.parquet`` file if
    available. Falls back to the pybaseball API when the raw file is missing
    or when ``from_api=True``.

    Parameters
    ----------
    year:
        Season year (2016-2025).
    out_dir:
        Directory to write ``statcast_agg_{year}.parquet``.
    force:
        If True, re-aggregate even if the agg file already exists.
    delay:
        Seconds to sleep after downloading from the API (courtesy rate-limit).
    min_bbe:
        Minimum batted-ball events per batter for inclusion.
    from_api:
        If True, always fetch from the pybaseball API even if a local raw
        file exists.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    if year not in _SEASON_DATES:
        raise ValueError(
            f"No season date range configured for {year}. "
            f"Supported years: {sorted(_SEASON_DATES)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statcast_agg_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping %d — file already exists at %s", year, out_path)
        return out_path

    # --- Try local raw file first (unless from_api is True) ---
    raw_path = out_dir / f"statcast_raw_{year}.parquet"
    if not from_api and raw_path.exists():
        logger.info(
            "Aggregating from local raw file for %d: %s", year, raw_path,
        )
        raw = pd.read_parquet(raw_path)
        agg = _aggregate_batter_statcast(raw, min_bbe=min_bbe)
        agg["season"] = year
        logger.info("  Aggregated to %d batters (min %d BBE)", len(agg), min_bbe)

        agg.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        size_mb = out_path.stat().st_size / 1_048_576
        logger.info("  Saved -> %s  (%.1f MB)", out_path, size_mb)
        return out_path

    if not from_api:
        logger.warning(
            "No local raw file for %d at %s — falling back to API",
            year, raw_path,
        )

    # --- API path ---
    try:
        from pybaseball import statcast
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for data fetching. Install with: uv sync"
        ) from exc

    start_dt, end_dt = _SEASON_DATES[year]
    logger.info("Fetching Statcast data for %d (%s -> %s) ...", year, start_dt, end_dt)

    raw = statcast(start_dt=start_dt, end_dt=end_dt)
    logger.info("  Downloaded %d pitch rows", len(raw))

    # Side-effect: save raw BBE data for future local aggregation
    bbe_for_raw = raw[
        raw["launch_speed"].notna()
        & raw["launch_angle"].notna()
        & raw["bb_type"].notna()
        & (raw["game_type"] == "R")
    ].copy()
    cols_present = [c for c in _RAW_KEEP_COLUMNS if c in bbe_for_raw.columns]
    bbe_for_raw = bbe_for_raw[cols_present].reset_index(drop=True)
    bbe_for_raw.to_parquet(raw_path, engine="pyarrow", compression="zstd", index=False)
    logger.info("  Saved raw BBE -> %s  (%d rows)", raw_path, len(bbe_for_raw))

    agg = _aggregate_batter_statcast(raw, min_bbe=min_bbe)
    agg["season"] = year
    logger.info("  Aggregated to %d batters (min %d BBE)", len(agg), min_bbe)

    agg.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved -> %s  (%.1f MB)", out_path, size_mb)

    if delay > 0:
        time.sleep(delay)

    return out_path


def _parse_season_tokens(tokens: list[str]) -> list[int]:
    """Parse season tokens into a sorted list of years."""
    years: list[int] = []
    for tok in tokens:
        if "-" in tok:
            start_s, end_s = tok.split("-", 1)
            years.extend(range(int(start_s), int(end_s) + 1))
        else:
            years.append(int(tok))
    return sorted(set(years))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Download Statcast data, aggregate per-batter metrics, save as Parquet.",
    )
    parser.add_argument(
        "--seasons", nargs="+", required=True, metavar="YEAR",
        help="Season(s) to download: individual years or a range (e.g. 2016-2025)",
    )
    parser.add_argument("--out-dir", default="data/raw", metavar="DIR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-api", action="store_true",
                        help="Force fetching from pybaseball API instead of local raw files")
    parser.add_argument("--delay", type=float, default=5.0, metavar="SEC")
    parser.add_argument("--min-bbe", type=int, default=50, metavar="N")
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to fetch: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            fetch_statcast(
                year, out_dir=args.out_dir, force=args.force,
                delay=args.delay, min_bbe=args.min_bbe,
                from_api=args.from_api,
            )
        except Exception:
            logger.exception("Failed to fetch %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

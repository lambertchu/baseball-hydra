"""Fetch and cache raw pitch-level Statcast batted-ball event data.

Downloads raw Statcast data for each season, filters to batted-ball events
(non-null launch_speed, launch_angle, bb_type, regular season only), and
saves a lean parquet file with only the columns needed for bucket analysis
and full-fidelity aggregation.

Usage
-----
    uv run python -m src.data.fetch_raw_statcast --seasons 2016-2025
    uv run python -m src.data.fetch_raw_statcast --seasons 2024 --force

Output
------
    data/raw/statcast_raw_{year}.parquet  (zstd-compressed)
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.data.fetch_statcast import _SEASON_DATES, _parse_season_tokens

logger = logging.getLogger(__name__)

_KEEP_COLUMNS = [
    "batter",
    "game_date",  # Required for weekly / ROS aggregation
    "launch_speed",
    "launch_angle",
    "bb_type",
    "events",
    "hc_x",
    "hc_y",
    "game_type",
    "woba_value",
    "estimated_woba_using_speedangle",
    # Columns needed for full-fidelity aggregation:
    "launch_speed_angle",  # barrel rate (value == 6)
    "estimated_ba_using_speedangle",  # xBA
    "estimated_slg_using_speedangle",  # xSLG
]


def fetch_raw_statcast(
    year: int,
    out_dir: str | Path = "data/raw",
    save_only_bbe_data: bool = True,
    force: bool = False,
    delay: float = 5.0,
) -> Path:
    """Download one season of raw Statcast BBE data and save as parquet.

    Parameters
    ----------
    year:
        Season year (must be in _SEASON_DATES).
    out_dir:
        Directory to write ``statcast_raw_{year}.parquet``.
    force:
        If True, re-download even if the file already exists.
    delay:
        Seconds to sleep after downloading (courtesy rate-limit).

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    try:
        import pybaseball
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for data fetching. Install with: uv sync"
        ) from exc

    pybaseball.cache.enable()

    if year not in _SEASON_DATES:
        raise ValueError(
            f"No season date range configured for {year}. "
            f"Supported years: {sorted(_SEASON_DATES)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statcast_raw_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping %d — file already exists at %s", year, out_path)
        return out_path

    start_dt, end_dt = _SEASON_DATES[year]
    logger.info(
        "Fetching raw Statcast data for %d (%s -> %s) ...", year, start_dt, end_dt
    )

    raw = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt)
    logger.info("  Downloaded %d pitch rows", len(raw))

    if not save_only_bbe_data:
        complete_out_path = out_dir / f"statcast_complete_{year}.parquet"
        raw.to_parquet(
            complete_out_path, engine="pyarrow", compression="zstd", index=False
        )
        size_mb = complete_out_path.stat().st_size / 1_048_576
        logger.info(
            "  Saved -> %s  (%.1f MB, %d rows)", complete_out_path, size_mb, len(raw)
        )

    else:
        # Filter to batted-ball events only
        bbe = raw[
            raw["launch_speed"].notna()
            & raw["launch_angle"].notna()
            & raw["bb_type"].notna()
            & (raw["game_type"] == "R")
        ].copy()
        logger.info("  Filtered to %d batted-ball events", len(bbe))

        # Keep only relevant columns (drop any that don't exist in the data)
        cols_present = [c for c in _KEEP_COLUMNS if c in bbe.columns]
        bbe = bbe[cols_present].reset_index(drop=True)

        bbe.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
        size_mb = out_path.stat().st_size / 1_048_576
        logger.info("  Saved -> %s  (%.1f MB, %d rows)", out_path, size_mb, len(bbe))

    if delay > 0:
        time.sleep(delay)

    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Download raw Statcast BBE data and save as Parquet.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        required=True,
        metavar="YEAR",
        help="Season(s) to download: individual years or a range (e.g. 2016-2025)",
    )
    parser.add_argument("--out-dir", default="data/raw", metavar="DIR")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if file exists"
    )
    parser.add_argument("--delay", type=float, default=5.0, metavar="SEC")
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to fetch: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            fetch_raw_statcast(
                year,
                out_dir=args.out_dir,
                force=args.force,
                delay=args.delay,
            )
        except Exception:
            logger.exception("Failed to fetch %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

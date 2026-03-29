"""Download all data sources for specified seasons.

Orchestrates the individual fetch modules (batting, statcast, speed, context)
into a single CLI command so users can download everything in one step.

Usage
-----
    python -m src.data.fetch_all --seasons 2016-2025
    python -m src.data.fetch_all --seasons 2024 2025 --out-dir data/raw --force
"""
from __future__ import annotations

import argparse
import logging
import time

from src.data.fetch_batting import _parse_season_tokens, fetch_batting
from src.data.fetch_context import fetch_park_factors, fetch_team_batting
from src.data.fetch_projections import fetch_all_projections
from src.data.fetch_raw_statcast import fetch_raw_statcast
from src.data.fetch_speed import fetch_bat_speed, fetch_sprint_speed
from src.data.fetch_statcast import fetch_statcast

logger = logging.getLogger(__name__)

# Number of data sources fetched per year (batting, raw_statcast, statcast,
# sprint speed, bat speed, park factors, team batting).
_SOURCES_PER_YEAR = 7


def fetch_all(
    years: list[int],
    out_dir: str = "data/raw",
    force: bool = False,
    delay: float = 3.0,
) -> dict[str, list[int]]:
    """Download all data sources for the given seasons.

    Parameters
    ----------
    years:
        Seasons to download.
    out_dir:
        Directory to write Parquet files into.
    force:
        Re-download even if cached files exist.
    delay:
        Seconds to wait between API calls.

    Returns
    -------
    dict
        Mapping of source name to list of years that failed.
    """
    failures: dict[str, list[int]] = {
        "batting": [],
        "raw_statcast": [],
        "statcast": [],
        "sprint_speed": [],
        "bat_speed": [],
        "park_factors": [],
        "team_batting": [],
    }

    total_years = len(years)
    for i, year in enumerate(years, 1):
        logger.info("=== Season %d  (%d/%d) ===", year, i, total_years)

        # Batting stats (FanGraphs)
        try:
            fetch_batting(year, out_dir=out_dir, force=force, delay=delay)
        except Exception:
            logger.exception("  FAILED: batting %d", year)
            failures["batting"].append(year)

        # Raw Statcast BBE data (before aggregation)
        try:
            fetch_raw_statcast(year, out_dir=out_dir, force=force, delay=delay)
        except Exception:
            logger.exception("  FAILED: raw_statcast %d", year)
            failures["raw_statcast"].append(year)

        # Statcast aggregated metrics (reads local raw file if available)
        try:
            fetch_statcast(
                year, out_dir=out_dir, force=force, delay=delay, min_bbe=50
            )
        except Exception:
            logger.exception("  FAILED: statcast %d", year)
            failures["statcast"].append(year)

        # Sprint speed
        try:
            fetch_sprint_speed(year, out_dir=out_dir, force=force, delay=delay)
        except Exception:
            logger.exception("  FAILED: sprint_speed %d", year)
            failures["sprint_speed"].append(year)

        # Bat speed (2024+ only; returns None for earlier years)
        try:
            fetch_bat_speed(year, out_dir=out_dir, force=force, delay=delay)
        except Exception:
            logger.exception("  FAILED: bat_speed %d", year)
            failures["bat_speed"].append(year)

        # Park factors
        try:
            fetch_park_factors(year, out_dir=out_dir, force=force, delay=delay)
        except Exception:
            logger.exception("  FAILED: park_factors %d", year)
            failures["park_factors"].append(year)

        # Team batting stats
        try:
            fetch_team_batting(year, out_dir=out_dir, force=force, delay=delay)
        except Exception:
            logger.exception("  FAILED: team_batting %d", year)
            failures["team_batting"].append(year)

    return failures


def fetch_all_with_projections(
    years: list[int],
    out_dir: str = "data/raw",
    force: bool = False,
    delay: float = 3.0,
    projection_systems: list[str] | None = None,
    ext_projections_dir: str = "data/external_projections",
) -> dict[str, list[int]]:
    """Download all data sources plus external projections for the latest year.

    Calls :func:`fetch_all` for per-year sources, then fetches external
    projections (Steamer, ZiPS, etc.) for the most recent year only.
    """
    failures = fetch_all(years, out_dir=out_dir, force=force, delay=delay)

    predict_year = max(years) + 1
    logger.info("=== Fetching external projections for %d ===", predict_year)
    failures["projections"] = []
    try:
        results = fetch_all_projections(
            predict_year,
            systems=projection_systems,
            out_dir=ext_projections_dir,
            force=force,
            delay=delay,
        )
        failed_systems = [s for s, p in results.items() if p is None]
        if failed_systems:
            logger.warning("  Failed projection systems: %s", failed_systems)
    except Exception:
        logger.exception("  FAILED: projections for %d", predict_year)
        failures["projections"].append(predict_year)

    return failures


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Download all data sources (batting, statcast, speed, context) for specified seasons.",
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
        "--force",
        action="store_true",
        help="Re-download even if cached Parquet files exist",
    )
    parser.add_argument("--delay", type=float, default=3.0, metavar="SEC")
    parser.add_argument(
        "--with-projections",
        action="store_true",
        help="Also fetch public projections (Steamer, ZiPS, etc.) for the prediction year",
    )
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Downloading all data for seasons: %s", years)
    start = time.monotonic()

    if args.with_projections:
        failures = fetch_all_with_projections(
            years, out_dir=args.out_dir, force=args.force, delay=args.delay
        )
    else:
        failures = fetch_all(
            years, out_dir=args.out_dir, force=args.force, delay=args.delay
        )

    elapsed = time.monotonic() - start
    total_failed = sum(len(v) for v in failures.values())

    logger.info("=" * 50)
    logger.info(
        "Done in %.0fs — %d seasons × %d sources",
        elapsed,
        len(years),
        _SOURCES_PER_YEAR,
    )
    if total_failed:
        logger.warning("%d failure(s):", total_failed)
        for source, yrs in failures.items():
            if yrs:
                logger.warning("  %s: %s", source, yrs)
    else:
        logger.info("All downloads succeeded.")


if __name__ == "__main__":
    main()

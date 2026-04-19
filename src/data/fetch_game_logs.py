"""Fetch per-(batter, ISO-week) batting stats for in-season / ROS analysis.

Downloads Baseball Reference weekly batting aggregates via
``pybaseball.batting_stats_range`` once per ISO week, concatenates them into
a single season-level parquet keyed by ``(mlbam_id, iso_year, iso_week)``.

BRef is used (not FanGraphs) because it publishes a date-range split endpoint
and includes ``mlbID`` — no extra ID lookup step is required. FanGraphs does
not expose arbitrary date ranges via pybaseball.

Usage
-----
    python -m src.data.fetch_game_logs --seasons 2016-2025

Output
------
    data/raw/batting_week_{year}.parquet  (zstd-compressed)
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.fetch_statcast import _SEASON_DATES, _parse_season_tokens

logger = logging.getLogger(__name__)

# BRef column → project-schema lowercase rename.
# Only a subset of BRef columns is retained; unknown columns are ignored.
_RENAME: dict[str, str] = {
    "mlbID": "mlbam_id",
    "Name": "name",
    "Age": "age",
    "Tm": "team",
    "Lev": "level",
    "G": "g",
    "PA": "pa",
    "AB": "ab",
    "R": "r",
    "H": "h",
    "2B": "doubles",
    "3B": "triples",
    "HR": "hr",
    "RBI": "rbi",
    "BB": "bb",
    "IBB": "ibb",
    "SO": "so",
    "HBP": "hbp",
    "SH": "sh",
    "SF": "sf",
    "GDP": "gdp",
    "SB": "sb",
    "CS": "cs",
    "BA": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
}

# Numeric columns that must be coerced from string → numeric (BRef returns strings).
_NUMERIC_COLS: tuple[str, ...] = (
    "age", "g", "pa", "ab", "r", "h", "doubles", "triples", "hr", "rbi",
    "bb", "ibb", "so", "hbp", "sh", "sf", "gdp", "sb", "cs",
    "avg", "obp", "slg", "ops",
)


def iso_weeks_in_season(
    year: int,
    season_dates: dict[int, tuple[str, str]] | None = None,
) -> list[tuple[int, int, date, date]]:
    """Enumerate ISO weeks that overlap the regular-season date range.

    Parameters
    ----------
    year:
        Season year.
    season_dates:
        Override ``_SEASON_DATES`` for testing.

    Returns
    -------
    list of (iso_year, iso_week, week_start_mon, week_end_sun) tuples.
    Monday-anchored ISO weeks fully covering the season span. First and last
    weeks may extend beyond the season window — callers should not assume a
    tuple's week_start equals the season's first game date.
    """
    dates = season_dates if season_dates is not None else _SEASON_DATES
    start_s, end_s = dates[year]
    start = date.fromisoformat(start_s)
    end = date.fromisoformat(end_s)

    # Snap start back to the Monday of its ISO week.
    week_start = start - timedelta(days=start.weekday())
    out: list[tuple[int, int, date, date]] = []
    cur = week_start
    while cur <= end:
        week_end = cur + timedelta(days=6)
        iso_year, iso_week, _ = cur.isocalendar()
        out.append((iso_year, iso_week, cur, week_end))
        cur = cur + timedelta(days=7)
    return out


def _normalize_bref_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename BRef columns to project schema and coerce numerics."""
    rename = {k: v for k, v in _RENAME.items() if k in df.columns}
    out = df.rename(columns=rename).copy()

    # Drop minor-league rows when present (Lev == 'Maj' for MLB).
    if "level" in out.columns:
        out = out[out["level"].astype(str).str.startswith("Maj")].copy()

    for col in _NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "mlbam_id" in out.columns:
        out["mlbam_id"] = pd.to_numeric(out["mlbam_id"], errors="coerce").astype("Int64")

    return out


def fetch_batter_weekly_stats(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 2.0,
    min_pa: int = 1,
) -> Path:
    """Fetch per-(batter, ISO-week) batting stats for one season and save.

    Makes one pybaseball call per ISO week in the season. Rate-limited via
    ``delay``. Idempotent: skips existing output unless ``force``.

    Parameters
    ----------
    year:
        Season year (must be in ``_SEASON_DATES``).
    out_dir:
        Directory to write ``batting_week_{year}.parquet``.
    force:
        If True, re-download even when the output exists.
    delay:
        Seconds to sleep between weekly queries (courtesy rate-limit).
    min_pa:
        Minimum PA per (batter, week) to retain (default 1; weeks with zero
        playing time are dropped naturally).

    Returns
    -------
    Path
        Path to the saved parquet file.
    """
    try:
        import pybaseball as pb
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for data fetching. Install with: uv sync"
        ) from exc

    if year not in _SEASON_DATES:
        raise ValueError(
            f"No season date range configured for {year}. "
            f"Supported years: {sorted(_SEASON_DATES)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batting_week_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping %d — file already exists at %s", year, out_path)
        return out_path

    weeks = iso_weeks_in_season(year)
    logger.info("Fetching %d ISO weeks for %d", len(weeks), year)

    frames: list[pd.DataFrame] = []
    for i, (iso_year, iso_week, wstart, wend) in enumerate(weeks, start=1):
        logger.info(
            "  [%d/%d] %d-W%02d  %s..%s",
            i, len(weeks), iso_year, iso_week, wstart, wend,
        )
        try:
            raw = pb.batting_stats_range(
                start_dt=wstart.isoformat(),
                end_dt=wend.isoformat(),
            )
        except Exception:
            logger.exception("  Failed week %d-W%02d", iso_year, iso_week)
            if delay > 0:
                time.sleep(delay)
            continue

        if raw is None or len(raw) == 0:
            logger.info("  (no rows)")
            continue

        df = _normalize_bref_columns(raw)
        df["iso_year"] = iso_year
        df["iso_week"] = iso_week
        df["week_start_date"] = pd.Timestamp(wstart)
        df["week_end_date"] = pd.Timestamp(wend)
        df["season"] = year

        if "pa" in df.columns:
            df = df[df["pa"].fillna(0) >= min_pa].copy()

        frames.append(df)

        if delay > 0:
            time.sleep(delay)

    if not frames:
        raise RuntimeError(f"No weekly data fetched for {year}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(
        out_path, engine="pyarrow", compression="zstd", index=False,
    )
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info(
        "  Saved -> %s  (%.1f MB, %d rows across %d weeks)",
        out_path, size_mb, len(combined), len(frames),
    )
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Download per-(batter, ISO-week) batting stats from BRef.",
    )
    parser.add_argument(
        "--seasons", nargs="+", required=True, metavar="YEAR",
        help="Season(s) to download: individual years or a range (e.g. 2016-2025)",
    )
    parser.add_argument("--out-dir", default="data/raw", metavar="DIR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay", type=float, default=2.0, metavar="SEC")
    parser.add_argument("--min-pa", type=int, default=1, metavar="PA")
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to fetch: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            fetch_batter_weekly_stats(
                year,
                out_dir=args.out_dir,
                force=args.force,
                delay=args.delay,
                min_pa=args.min_pa,
            )
        except Exception:
            logger.exception("Failed to fetch %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

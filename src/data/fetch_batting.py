"""Fetch and cache FanGraphs batting stats via pybaseball.

Usage
-----
    python -m src.data.fetch_batting --seasons 2016-2025 --out-dir data/raw

Output
------
    data/raw/batting_{year}.parquet  (zstd-compressed)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Columns to keep from FanGraphs batting_stats response.
# pybaseball returns many columns; we select the ones relevant for modeling.
_KEEP_COLS: list[str] = [
    "IDfg",
    "Name",
    "Age",
    "Team",
    "G",
    "PA",
    "AB",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "R",
    "RBI",
    "BB",
    "IBB",
    "SO",
    "HBP",
    "SF",
    "SH",
    "SB",
    "CS",
    "AVG",
    "OBP",
    "SLG",
    "OPS",
    "BABIP",
    "wOBA",
    "wRC+",
    "WAR",
]

# Rename to lowercase project schema
_RENAME: dict[str, str] = {
    "IDfg": "idfg",
    "Name": "name",
    "Age": "age",
    "Team": "team",
    "G": "g",
    "PA": "pa",
    "AB": "ab",
    "H": "h",
    "1B": "singles",
    "2B": "doubles",
    "3B": "triples",
    "HR": "hr",
    "R": "r",
    "RBI": "rbi",
    "BB": "bb",
    "IBB": "ibb",
    "SO": "so",
    "HBP": "hbp",
    "SF": "sf",
    "SH": "sh",
    "SB": "sb",
    "CS": "cs",
    "AVG": "avg",
    "OBP": "obp",
    "SLG": "slg",
    "OPS": "ops",
    "BABIP": "babip",
    "wOBA": "woba",
    "wRC+": "wrc_plus",
    "WAR": "war",
}


def fetch_batting(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 3.0,
    min_pa: int = 0,
) -> Path:
    """Download one season of FanGraphs batting stats and save to Parquet.

    Parameters
    ----------
    year:
        Season year (2016-2025).
    out_dir:
        Directory to write ``batting_{year}.parquet``.
    force:
        If True, re-download even if the file already exists.
    delay:
        Seconds to sleep after downloading (courtesy rate-limit).
    min_pa:
        Minimum plate appearances to include (0 = all qualified + unqualified).

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    try:
        import pybaseball as pb
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for data fetching. "
            "Install with: uv sync"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"batting_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping %d — file already exists at %s", year, out_path)
        return out_path

    logger.info("Fetching FanGraphs batting stats for %d …", year)
    raw = pb.batting_stats(start_season=year, end_season=year, qual=0)
    logger.info("  Downloaded %d rows", len(raw))

    # Filter by minimum PA
    if min_pa > 0:
        raw = raw[raw["PA"] >= min_pa].copy()
        logger.info("  After PA >= %d filter: %d rows", min_pa, len(raw))

    # Keep only relevant columns (some may not exist in all years)
    cols = [c for c in _KEEP_COLS if c in raw.columns]
    df = raw[cols].copy()

    # Rename to project schema
    rename = {k: v for k, v in _RENAME.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Add season column
    df["season"] = year

    # Ensure numeric types for key columns
    for col in ["idfg", "pa", "ab", "hr", "r", "rbi", "sb", "cs", "bb", "so", "hbp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved → %s  (%.1f MB, %d rows)", out_path, size_mb, len(df))

    if delay > 0:
        time.sleep(delay)

    return out_path


def _parse_season_tokens(tokens: list[str]) -> list[int]:
    """Parse season tokens into a sorted list of years.

    Accepts individual years (``"2022"``) or inclusive ranges (``"2016-2025"``).
    """
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
        description="Download FanGraphs batting stats and save as Parquet.",
    )
    parser.add_argument(
        "--seasons", nargs="+", required=True, metavar="YEAR",
        help="Season(s) to download: individual years or a range (e.g. 2016-2025)",
    )
    parser.add_argument("--out-dir", default="data/raw", metavar="DIR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0, metavar="SEC")
    parser.add_argument("--min-pa", type=int, default=0, metavar="PA")
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to fetch: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            fetch_batting(
                year, out_dir=args.out_dir, force=args.force,
                delay=args.delay, min_pa=args.min_pa,
            )
        except Exception:
            logger.exception("Failed to fetch %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

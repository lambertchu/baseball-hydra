"""Fetch and cache public projection systems from FanGraphs.

Downloads projections from Steamer, ZiPS, The Bat, and The Bat X via the
FanGraphs JSON API.  These are used for **comparison only** and never enter
the training pipeline.

Usage
-----
    python -m src.data.fetch_projections --year 2026
    python -m src.data.fetch_projections --year 2026 --systems steamer zips

Output
------
    data/external_projections/{system}_{year}.csv
"""
from __future__ import annotations

import argparse
import datetime
import io
import json
import logging
import time
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Supported projection systems and their FanGraphs API type parameter.
PROJECTION_SYSTEMS: dict[str, str] = {
    "steamer": "steamer",
    "zips": "zips",
    "thebat": "thebat",
    "thebatx": "thebatx",
}

# Short display names used in console output.
DISPLAY_NAMES: dict[str, str] = {
    "steamer": "Stmr",
    "zips": "ZiPS",
    "thebat": "Bat",
    "thebatx": "BatX",
}

_FANGRAPHS_API_URL = (
    "https://www.fangraphs.com/api/projections"
    "?type={system}&stats=bat&pos=all&team=0&players=0&lg=all"
)

_FANGRAPHS_HTML_URL = (
    "https://www.fangraphs.com/projections"
    "?type={system}&stats=bat&pos=all&team=0&players=0"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.fangraphs.com/projections",
}

# Columns we keep from the projection response.
_KEEP_COLS = ["idfg", "name", "pa", "obp", "slg", "hr", "r", "rbi", "sb"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and filter columns to the project schema."""
    # Build case-insensitive rename map from whatever columns exist.
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    # Apply specific renames for known FanGraphs column names.
    rename = {}
    if "playerid" in df.columns:
        rename["playerid"] = "idfg"
    if "playername" in df.columns:
        rename["playername"] = "name"
    if rename:
        df = df.rename(columns=rename)

    # Keep only the columns we need (if present).
    available = [c for c in _KEEP_COLS if c in df.columns]
    return df[available].copy()


def _fetch_projections_api(system: str, year: int) -> pd.DataFrame:
    """Try fetching projections from the FanGraphs JSON API."""
    url = _FANGRAPHS_API_URL.format(system=system)
    if year:
        url += f"&season={year}"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    if not raw:
        raise ValueError(f"Empty API response for {system} {year}")
    return pd.DataFrame(raw)


def _fetch_projections_html(system: str, year: int) -> pd.DataFrame:
    """Fallback: scrape the HTML projections page for a table."""
    url = _FANGRAPHS_HTML_URL.format(system=system)
    if year:
        url += f"&season={year}"
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        raise ValueError(f"No HTML tables found for {system} {year}")
    # Take the largest table (the main projections table).
    return max(tables, key=len)


def fetch_projections(
    system: str,
    year: int,
    out_dir: str | Path = "data/external_projections",
    force: bool = False,
    delay: float = 3.0,
) -> Path | None:
    """Download projections for one system/year and save to CSV.

    Parameters
    ----------
    system:
        Projection system key (e.g. ``"steamer"``).
    year:
        Season year for projections.
    out_dir:
        Output directory.
    force:
        Re-download even if file exists.
    delay:
        Seconds to sleep after downloading.

    Returns
    -------
    Path | None
        Path to saved CSV file, or None on failure.
    """
    if system not in PROJECTION_SYSTEMS:
        logger.error("Unknown projection system: %s", system)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{system}_{year}.csv"

    if out_path.exists() and not force:
        logger.info("Skipping %s %d projections — file exists at %s", system, year, out_path)
        return out_path

    current_year = datetime.date.today().year
    if year != current_year:
        logger.warning(
            "FanGraphs projections API only serves current-year data. "
            "Requesting %d but API will return %d projections.",
            year,
            current_year,
        )

    logger.info("Fetching %s projections for %d …", system, year)

    df: pd.DataFrame | None = None

    # Try JSON API first.
    try:
        raw = _fetch_projections_api(system, year)
        logger.info("  API returned %d rows for %s", len(raw), system)
        df = raw
    except Exception as exc:
        logger.warning("  API fetch failed for %s %d: %s — trying HTML fallback", system, year, exc)

    # Fallback to HTML scrape.
    if df is None:
        try:
            raw = _fetch_projections_html(system, year)
            logger.info("  HTML scrape returned %d rows for %s", len(raw), system)
            df = raw
        except Exception as exc:
            logger.warning("  HTML fallback also failed for %s %d: %s", system, year, exc)
            return None

    # Normalize columns.
    df = _normalize_columns(df)

    if "idfg" not in df.columns:
        logger.warning("  No player ID column found for %s %d — skipping", system, year)
        return None

    # Ensure numeric types.
    df["idfg"] = pd.to_numeric(df["idfg"], errors="coerce")
    df = df.dropna(subset=["idfg"])
    df["idfg"] = df["idfg"].astype(int)
    for col in ["pa", "obp", "slg", "hr", "r", "rbi", "sb"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Deduplicate.
    df = df.drop_duplicates(subset=["idfg"], keep="first")

    # Add metadata.
    df["season"] = year
    df["projection_system"] = system

    df.to_csv(out_path, index=False)
    size_kb = out_path.stat().st_size / 1024
    logger.info("  Saved → %s  (%.1f KB, %d players)", out_path, size_kb, len(df))

    if delay > 0:
        time.sleep(delay)

    return out_path


def fetch_all_projections(
    year: int,
    systems: list[str] | None = None,
    out_dir: str | Path = "data/external_projections",
    force: bool = False,
    delay: float = 3.0,
) -> dict[str, Path | None]:
    """Download projections for all specified systems.

    Returns
    -------
    dict
        Mapping system name → path (or None on failure).
    """
    if systems is None:
        systems = list(PROJECTION_SYSTEMS.keys())

    results: dict[str, Path | None] = {}
    for system in systems:
        results[system] = fetch_projections(
            system, year, out_dir=out_dir, force=force, delay=delay,
        )
    return results


def load_projections(
    year: int,
    systems: list[str] | None = None,
    out_dir: str | Path = "data/external_projections",
) -> pd.DataFrame | None:
    """Load cached projection CSVs and concatenate.

    Returns
    -------
    pd.DataFrame | None
        Combined DataFrame with ``projection_system`` column, or None if
        no projection files are found.
    """
    if systems is None:
        systems = list(PROJECTION_SYSTEMS.keys())

    out_dir = Path(out_dir)
    frames: list[pd.DataFrame] = []
    for system in systems:
        path = out_dir / f"{system}_{year}.csv"
        if path.exists():
            df = pd.read_csv(path)
            # Normalize raw FanGraphs columns if needed (e.g. manually
            # downloaded CSVs that weren't saved via fetch_projections).
            if "idfg" not in df.columns:
                df = _normalize_columns(df)
                if "idfg" in df.columns:
                    df["idfg"] = pd.to_numeric(df["idfg"], errors="coerce")
                    df = df.dropna(subset=["idfg"])
                    df["idfg"] = df["idfg"].astype(int)
                    for col in ["pa", "obp", "slg", "hr", "r", "rbi", "sb"]:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.drop_duplicates(subset=["idfg"], keep="first")
            if "projection_system" not in df.columns:
                df["projection_system"] = system
            frames.append(df)
            logger.info("Loaded %s projections from %s", system, path)
        else:
            logger.debug("No cached projections for %s %d", system, year)

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Download FanGraphs public projections and save as CSV.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Season year for projections (default: 2026)",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(PROJECTION_SYSTEMS.keys()),
        choices=list(PROJECTION_SYSTEMS.keys()),
        help="Projection systems to fetch (default: all)",
    )
    parser.add_argument("--out-dir", default="data/external_projections", metavar="DIR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0, metavar="SEC")
    args = parser.parse_args()

    results = fetch_all_projections(
        args.year,
        systems=args.systems,
        out_dir=args.out_dir,
        force=args.force,
        delay=args.delay,
    )

    succeeded = sum(1 for v in results.values() if v is not None)
    failed = sum(1 for v in results.values() if v is None)
    logger.info("Done: %d succeeded, %d failed", succeeded, failed)
    if failed:
        for sys_name, path in results.items():
            if path is None:
                logger.warning("  Failed: %s", sys_name)


if __name__ == "__main__":
    main()

"""Fetch and cache ballpark factors and team-level batting stats.

Ballpark factors are scraped from FanGraphs (via pybaseball). Team batting
stats are fetched from FanGraphs team-level data.

Usage
-----
    python -m src.data.fetch_context --seasons 2016-2025 --out-dir data/raw

Output
------
    data/raw/park_factors_{year}.parquet   (zstd-compressed)
    data/raw/team_batting_{year}.parquet   (zstd-compressed)
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_park_factors(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 3.0,
) -> Path:
    """Download FanGraphs park factors for one season and save to Parquet.

    Parameters
    ----------
    year:
        Season year.
    out_dir:
        Output directory.
    force:
        Re-download even if file exists.
    delay:
        Seconds to sleep after downloading.

    Returns
    -------
    Path
        Path to saved Parquet file.
    """
    try:
        from pybaseball import park_factors as pb_park_factors
    except ImportError:
        # Fallback: try the general FanGraphs park factors scraper
        pb_park_factors = None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"park_factors_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping park factors %d — file exists at %s", year, out_path)
        return out_path

    logger.info("Fetching park factors for %d …", year)

    if pb_park_factors is not None:
        try:
            raw = pb_park_factors(year)
            logger.info("  Downloaded %d rows from pybaseball.park_factors", len(raw))
        except Exception:
            logger.warning("  pybaseball.park_factors failed, using fallback")
            raw = _build_fallback_park_factors(year)
    else:
        raw = _build_fallback_park_factors(year)

    df = _normalize_park_factors(raw)
    df["season"] = year

    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved → %s  (%.1f MB, %d parks)", out_path, size_mb, len(df))

    if delay > 0:
        time.sleep(delay)

    return out_path


def _normalize_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize park factors DataFrame to a standard schema.

    Target columns: team, park_factor_runs, park_factor_hr
    Park factors are on a 100-scale (100 = neutral). We normalize to a
    multiplier (1.0 = neutral).
    """
    result = pd.DataFrame()

    # Try to find the team column
    for col in ["Team", "team", "Tm", "tm"]:
        if col in df.columns:
            result["team"] = df[col]
            break

    if "team" not in result.columns:
        result["team"] = df.iloc[:, 0] if len(df.columns) > 0 else "UNK"

    # Park factor for runs
    for col in ["Basic", "basic", "R", "Runs", "PF", "pf", "Factor"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            # If values are on 100-scale, convert to multiplier
            if vals.median() > 50:
                vals = vals / 100.0
            result["park_factor_runs"] = vals
            break

    if "park_factor_runs" not in result.columns:
        result["park_factor_runs"] = 1.0

    # Park factor for home runs
    for col in ["HR", "hr", "Home Run", "home_run"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.median() > 50:
                vals = vals / 100.0
            result["park_factor_hr"] = vals
            break

    if "park_factor_hr" not in result.columns:
        result["park_factor_hr"] = result["park_factor_runs"]

    return result


def _build_fallback_park_factors(year: int) -> pd.DataFrame:
    """Build fallback park factors using known approximate values.

    These are rough multi-year averages. Used when the pybaseball API
    is unavailable.
    """
    # Approximate park factors (runs, 100-scale) for MLB parks
    _PARK_FACTORS: dict[str, tuple[int, int]] = {
        "COL": (115, 120), "BOS": (104, 105), "CIN": (104, 108),
        "TEX": (103, 106), "CHC": (102, 110), "PHI": (102, 104),
        "MIL": (101, 107), "BAL": (101, 108), "MIN": (101, 103),
        "ATL": (101, 103), "TOR": (100, 103), "CLE": (100, 100),
        "LAA": (100, 98), "WSH": (100, 99), "NYY": (100, 115),
        "HOU": (99, 101), "DET": (99, 93), "ARI": (99, 101),
        "KC": (99, 96), "STL": (98, 99), "PIT": (98, 93),
        "CHW": (98, 101), "SEA": (97, 95), "SD": (97, 90),
        "TB": (97, 95), "LAD": (97, 94), "NYM": (97, 100),
        "SF": (96, 88), "MIA": (95, 88), "OAK": (96, 94),
    }
    rows = []
    for team, (pf_runs, pf_hr) in _PARK_FACTORS.items():
        rows.append({"Team": team, "Basic": pf_runs, "HR": pf_hr})
    return pd.DataFrame(rows)


def fetch_team_batting(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 3.0,
) -> Path:
    """Download team-level batting stats for one season and save to Parquet.

    Parameters
    ----------
    year:
        Season year.
    out_dir:
        Output directory.
    force:
        Re-download even if file exists.
    delay:
        Seconds to sleep after downloading.

    Returns
    -------
    Path
        Path to saved Parquet file.
    """
    try:
        import pybaseball as pb
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for data fetching. Install with: uv sync"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"team_batting_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping team batting %d — file exists at %s", year, out_path)
        return out_path

    logger.info("Fetching team batting stats for %d …", year)
    raw = pb.team_batting(start_season=year, end_season=year)
    logger.info("  Downloaded %d teams", len(raw))

    df = _normalize_team_batting(raw, year)

    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved → %s  (%.1f MB)", out_path, size_mb)

    if delay > 0:
        time.sleep(delay)

    return out_path


def _normalize_team_batting(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """Normalize team batting stats to project schema."""
    df = pd.DataFrame()

    # Team abbreviation
    for col in ["Tm", "Team", "team", "teamIDfg", "abbreviation"]:
        if col in raw.columns:
            df["team"] = raw[col]
            break
    if "team" not in df.columns and len(raw.columns) > 0:
        df["team"] = raw.iloc[:, 0]

    # Games played (for per-game normalization)
    # Games played (for per-game normalization)
    num_games = 60 if year == 2020 else 162
    if "G" in raw.columns:
        g = pd.to_numeric(raw["G"], errors="coerce").fillna(num_games)
    else:
        g = pd.Series([num_games] * len(raw))

    # Team runs scored per game
    if "R" in raw.columns:
        df["team_runs_per_game"] = pd.to_numeric(raw["R"], errors="coerce") / g
    else:
        df["team_runs_per_game"] = 4.5  # league average fallback

    # Team OPS
    if "OPS" in raw.columns:
        df["team_ops"] = pd.to_numeric(raw["OPS"], errors="coerce")
    elif "OBP" in raw.columns and "SLG" in raw.columns:
        df["team_ops"] = (
            pd.to_numeric(raw["OBP"], errors="coerce")
            + pd.to_numeric(raw["SLG"], errors="coerce")
        )
    else:
        df["team_ops"] = 0.720  # league average fallback

    # Team stolen bases (total)
    if "SB" in raw.columns:
        df["team_sb"] = pd.to_numeric(raw["SB"], errors="coerce")
    else:
        df["team_sb"] = 80  # rough league average fallback

    # Team stolen bases per game (normalized)
    df["team_sb_per_game"] = df["team_sb"] / g

    df["season"] = year
    return df


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
        description="Download park factors and team batting stats, save as Parquet.",
    )
    parser.add_argument(
        "--seasons", nargs="+", required=True, metavar="YEAR",
        help="Season(s) to download: individual years or a range (e.g. 2016-2025)",
    )
    parser.add_argument("--out-dir", default="data/raw", metavar="DIR")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0, metavar="SEC")
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to fetch: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            fetch_park_factors(year, out_dir=args.out_dir, force=args.force, delay=args.delay)
            fetch_team_batting(year, out_dir=args.out_dir, force=args.force, delay=args.delay)
        except Exception:
            logger.exception("Failed to fetch context data for %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

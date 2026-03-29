"""Fetch and cache sprint speed and bat speed data from Baseball Savant.

Sprint speed has been tracked since 2015. Bat speed is only available from
2024 onward. Both are accessed via pybaseball's statcast leaderboard APIs.

Usage
-----
    python -m src.data.fetch_speed --seasons 2016-2025 --out-dir data/raw

Output
------
    data/raw/sprint_speed_{year}.parquet  (zstd-compressed)
    data/raw/bat_speed_{year}.parquet     (2024+ only)
"""
from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Bat speed data is only available from 2024.
_BAT_SPEED_FIRST_YEAR = 2024


def fetch_sprint_speed(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 3.0,
) -> Path:
    """Download sprint speed leaderboard for one season and save to Parquet.

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
        from pybaseball import statcast_sprint_speed
    except ImportError as exc:
        raise ImportError(
            "pybaseball is required for data fetching. Install with: uv sync"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sprint_speed_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping sprint speed %d — file exists at %s", year, out_path)
        return out_path

    logger.info("Fetching sprint speed for %d …", year)
    raw = statcast_sprint_speed(year, min_opp=0)
    logger.info("  Downloaded %d rows", len(raw))

    # Keep key columns and rename to project schema
    keep = {}
    if "player_id" in raw.columns:
        keep["player_id"] = "mlbam_id"
    elif "entity_id" in raw.columns:
        keep["entity_id"] = "mlbam_id"
    if "sprint_speed" in raw.columns:
        keep["sprint_speed"] = "sprint_speed"
    elif "hp_to_1b" in raw.columns:
        keep["hp_to_1b"] = "sprint_speed"

    if not keep:
        logger.warning("  Unexpected columns in sprint speed data: %s", list(raw.columns))
        # Fallback: save all columns
        df = raw.copy()
    else:
        available = {k: v for k, v in keep.items() if k in raw.columns}
        df = raw[list(available.keys())].copy()
        df = df.rename(columns=available)

    df["season"] = year

    # Ensure mlbam_id is numeric
    if "mlbam_id" in df.columns:
        df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
        df = df.dropna(subset=["mlbam_id"])
        df["mlbam_id"] = df["mlbam_id"].astype(int)

    # Deduplicate (take first/best entry per player)
    if "mlbam_id" in df.columns:
        df = df.drop_duplicates(subset=["mlbam_id"], keep="first")

    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved → %s  (%.1f MB, %d players)", out_path, size_mb, len(df))

    if delay > 0:
        time.sleep(delay)

    return out_path


_BAT_TRACKING_URL = (
    "https://baseballsavant.mlb.com/leaderboard/bat-tracking"
    "?attackZone=&batSide=&contactType=&count=&dateStart=&dateEnd="
    "&gameType=&isHardHit=&minSwings=0&minGroupSwings=1"
    "&pitchHand=&pitchType=&playerType=Batter&season={year}"
    "&team=&type=&csv=true"
)


def _fetch_bat_speed_leaderboard(year: int) -> pd.DataFrame:
    """Fetch bat tracking leaderboard CSV from Baseball Savant."""
    import io
    import urllib.request

    url = _BAT_TRACKING_URL.format(year=year)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return pd.read_csv(io.BytesIO(resp.read()))


def fetch_bat_speed(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 3.0,
) -> Path | None:
    """Download bat speed leaderboard for one season and save to Parquet.

    Bat speed data is only available from 2024 onward. Returns None for
    earlier years.

    Parameters
    ----------
    year:
        Season year (2024+).
    out_dir:
        Output directory.
    force:
        Re-download even if file exists.
    delay:
        Seconds to sleep after downloading.

    Returns
    -------
    Path | None
        Path to saved Parquet file, or None if year < 2024.
    """
    if year < _BAT_SPEED_FIRST_YEAR:
        logger.info("Skipping bat speed %d — data only available from %d+", year, _BAT_SPEED_FIRST_YEAR)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bat_speed_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping bat speed %d — file exists at %s", year, out_path)
        return out_path

    logger.info("Fetching bat speed for %d …", year)
    try:
        raw = _fetch_bat_speed_leaderboard(year)
    except Exception:
        logger.warning("  Bat speed fetch failed for %d — API may not support this year", year)
        return None

    logger.info("  Downloaded %d rows", len(raw))

    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    norm_to_orig = {_norm(c): c for c in raw.columns}

    def _pick(*candidates: str) -> str | None:
        for c in candidates:
            if c in norm_to_orig:
                return norm_to_orig[c]
        return None

    rename_map: dict[str, str] = {}

    id_col = _pick("id", "player_id", "batter_id", "entity_id", "mlbam_id")
    if id_col is not None:
        rename_map[id_col] = "mlbam_id"

    canonical_map = {
        "avg_bat_speed": ["avg_bat_speed", "bat_speed"],
        "avg_swing_speed": ["avg_swing_speed", "swing_speed"],
        "squared_up_rate": ["squared_up_rate", "squared_up_pct", "squared_up_percentage"],
        "blast_rate": ["blast_rate", "blast_pct", "blast_percentage"],
        "fast_swing_rate": ["fast_swing_rate", "fast_swing_pct", "fast_swing_percentage"],
        "bat_tracking_swings": ["swings", "tracked_swings", "swing_count", "total_swings"],
        "bat_tracking_bbe": ["bbe", "batted_ball_events", "tracked_bbe"],
        "bat_tracking_blasts": ["blasts", "blast_count"],
        "bat_tracking_squared_up": ["squared_ups", "squared_up_count"],
        "bat_tracking_fast_swings": ["fast_swings", "fast_swing_count"],
    }

    for canonical, candidates in canonical_map.items():
        col = _pick(*candidates)
        if col is not None:
            rename_map[col] = canonical

    df = raw.rename(columns=rename_map).copy()
    df["season"] = year

    # Keep known canonical columns if available
    target_cols = [
        "mlbam_id",
        "season",
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
    available = [c for c in target_cols if c in df.columns]
    df = df[available].copy()

    if "mlbam_id" in df.columns:
        df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
        df = df.dropna(subset=["mlbam_id"])
        df["mlbam_id"] = df["mlbam_id"].astype(int)
        df = df.drop_duplicates(subset=["mlbam_id"], keep="first")

    for col in df.columns:
        if col in {"mlbam_id", "season"}:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved → %s  (%.1f MB, %d players)", out_path, size_mb, len(df))

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
        description="Download sprint speed and bat speed data, save as Parquet.",
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
            fetch_sprint_speed(year, out_dir=args.out_dir, force=args.force, delay=args.delay)
            fetch_bat_speed(year, out_dir=args.out_dir, force=args.force, delay=args.delay)
        except Exception:
            logger.exception("Failed to fetch speed data for %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

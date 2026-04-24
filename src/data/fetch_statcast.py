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
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# MLB regular-season date ranges (opening day → final regular-season game).
# These intentionally EXCLUDE the postseason — Wild Card / Division /
# Championship / World Series contamination was producing inflated ytd
# counts for playoff teams and leaking postseason games into the weekly
# snapshot pipeline's ROS targets.
#
# Statcast's ``fetch_statcast`` pipeline also filters on ``game_type == 'R'``
# in ``_filter_bbe``, but BRef's ``batting_stats_range`` endpoint used by
# ``fetch_game_logs`` does not — so clamping the date window here is the
# only robust guard against postseason bleed-through for the in-season
# pipeline.
#
# For the in-progress season, the end date is derived from ``date.today()``
# at import time so fetchers never hit future-dated empty ranges on BRef /
# Statcast, and refreshes automatically pull new weeks as they complete.
# A conservative upper cap (late September) guards against postseason
# bleed-through if the module is re-imported after the regular season ends
# without the entry being updated to a concrete end date.
_CURRENT_SEASON: int = 2026
_CURRENT_SEASON_REG_END_CAP: str = "2026-09-27"
_IN_PROGRESS_SEASON_END: str = min(
    _date.today().isoformat(), _CURRENT_SEASON_REG_END_CAP
)

_SEASON_DATES: dict[int, tuple[str, str]] = {
    2016: ("2016-04-03", "2016-10-02"),
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-28", "2019-09-29"),
    2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-27", "2025-09-28"),
    _CURRENT_SEASON: ("2026-03-26", _IN_PROGRESS_SEASON_END),
}


def _filter_bbe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter raw Statcast pitch data to regular-season batted-ball events."""
    return df[
        df["launch_speed"].notna()
        & df["launch_angle"].notna()
        & df["bb_type"].notna()
        & (df["game_type"] == "R")
    ].copy()


def _compute_bbe_metrics(
    bbe: pd.DataFrame,
    group_keys: list[str],
    min_bbe: int,
    extra_aggs: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Compute per-group Statcast quality metrics on filtered BBE rows.

    Shared by season-level and weekly aggregators — the only differences are
    the group_keys and optional extras (e.g. weekly adds `week_start_date`).

    Parameters
    ----------
    bbe:
        Already-filtered BBE DataFrame (``_filter_bbe`` output).
    group_keys:
        Columns to group by. ``["batter"]`` for season, or
        ``["batter", "iso_year", "iso_week"]`` for weekly.
    min_bbe:
        Minimum BBE per group for inclusion.
    extra_aggs:
        Optional extra columns mapped to ``(src_col, agg_method_name)``, e.g.
        ``{"week_start_date": ("game_date", "min")}``.
    """
    if bbe.empty:
        return pd.DataFrame()

    grouped = bbe.groupby(group_keys)

    agg = pd.DataFrame({
        "bbe_count": grouped.size(),
        "avg_exit_velocity": grouped["launch_speed"].mean(),
        "ev_p95": grouped["launch_speed"].quantile(0.95),
        "max_exit_velocity": grouped["launch_speed"].max(),
        "avg_launch_angle": grouped["launch_angle"].mean(),
    })

    if extra_aggs:
        for dst, (src, method) in extra_aggs.items():
            agg[dst] = getattr(grouped[src], method)()

    for col in (
        "estimated_woba_using_speedangle",
        "estimated_ba_using_speedangle",
        "estimated_slg_using_speedangle",
    ):
        agg[col] = grouped[col].mean() if col in bbe.columns else np.nan

    if "launch_speed_angle" in bbe.columns:
        barrels = bbe[bbe["launch_speed_angle"] == 6].groupby(group_keys).size()
        agg["barrel_rate"] = (barrels / agg["bbe_count"]).fillna(0.0)
    else:
        agg["barrel_rate"] = np.nan

    hard_hits = bbe[bbe["launch_speed"] >= 95.0].groupby(group_keys).size()
    agg["hard_hit_rate"] = (hard_hits / agg["bbe_count"]).fillna(0.0)

    sweet = (
        bbe[(bbe["launch_angle"] >= 8) & (bbe["launch_angle"] <= 32)]
        .groupby(group_keys)
        .size()
    )
    agg["sweet_spot_rate"] = (sweet / agg["bbe_count"]).fillna(0.0)

    agg = agg[agg["bbe_count"] >= min_bbe].reset_index()
    return agg.rename(columns={"batter": "mlbam_id"})


def _aggregate_batter_statcast(df: pd.DataFrame, min_bbe: int = 50) -> pd.DataFrame:
    """Aggregate raw Statcast pitch data into per-batter-season quality metrics."""
    return _compute_bbe_metrics(_filter_bbe(df), ["batter"], min_bbe)


def _aggregate_batter_statcast_weekly(
    df: pd.DataFrame,
    min_bbe: int = 5,
) -> pd.DataFrame:
    """Aggregate raw Statcast pitch data into per-(batter, ISO-week) quality metrics.

    Uses a lower default ``min_bbe`` than the season aggregator because a
    single week has ~15-25 BBE for a regular starter.
    """
    if "game_date" not in df.columns:
        raise ValueError(
            "Weekly aggregation requires a 'game_date' column. "
            "Re-fetch raw Statcast with the updated _KEEP_COLUMNS."
        )

    bbe = _filter_bbe(df)
    if bbe.empty:
        return pd.DataFrame()

    bbe["game_date"] = pd.to_datetime(bbe["game_date"])
    iso = bbe["game_date"].dt.isocalendar()
    bbe["iso_year"] = iso["year"].astype(int)
    bbe["iso_week"] = iso["week"].astype(int)

    return _compute_bbe_metrics(
        bbe,
        group_keys=["batter", "iso_year", "iso_week"],
        min_bbe=min_bbe,
        extra_aggs={"week_start_date": ("game_date", "min")},
    )


# Columns to keep when saving raw BBE data as a side-effect of API fetches.
# Defined locally (not imported from fetch_raw_statcast) to avoid circular imports.
_RAW_KEEP_COLUMNS = [
    "batter",
    "game_date",
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


def fetch_statcast_weekly(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    min_bbe: int = 5,
) -> Path:
    """Build per-(batter, iso-week) Statcast aggregates from a cached raw file.

    Reads ``statcast_raw_{year}.parquet`` (which must contain ``game_date``),
    aggregates by ISO week, and writes ``statcast_agg_week_{year}.parquet``.

    Unlike :func:`fetch_statcast`, this function does not fall back to the API:
    it requires the raw cache to already be present. Run
    ``python -m src.data.fetch_raw_statcast --seasons {year}`` first if needed.

    Parameters
    ----------
    year:
        Season year (2016-2025).
    out_dir:
        Directory containing ``statcast_raw_{year}.parquet`` and receiving
        the weekly aggregated output.
    force:
        If True, rebuild even if the weekly agg file already exists.
    min_bbe:
        Minimum batted-ball events per (batter, week) for inclusion.

    Returns
    -------
    Path
        Path to the saved ``statcast_agg_week_{year}.parquet``.
    """
    if year not in _SEASON_DATES:
        raise ValueError(
            f"No season date range configured for {year}. "
            f"Supported years: {sorted(_SEASON_DATES)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statcast_agg_week_{year}.parquet"

    if out_path.exists() and not force:
        logger.info("Skipping %d — weekly agg exists at %s", year, out_path)
        return out_path

    raw_path = out_dir / f"statcast_raw_{year}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw Statcast file missing: {raw_path}. "
            f"Run: python -m src.data.fetch_raw_statcast --seasons {year}"
        )

    logger.info("Aggregating weekly Statcast for %d from %s", year, raw_path)
    raw = pd.read_parquet(raw_path)
    if "game_date" not in raw.columns:
        raise ValueError(
            f"{raw_path} lacks 'game_date' — re-fetch raw with updated "
            f"_KEEP_COLUMNS: python -m src.data.fetch_raw_statcast "
            f"--seasons {year} --force"
        )

    agg = _aggregate_batter_statcast_weekly(raw, min_bbe=min_bbe)
    agg["season"] = year
    logger.info(
        "  Aggregated to %d (batter, week) rows (min %d BBE)", len(agg), min_bbe,
    )

    agg.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info("  Saved -> %s  (%.1f MB)", out_path, size_mb)
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

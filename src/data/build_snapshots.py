"""Build per-(player, season, ISO-week) snapshots for in-season modeling.

Combines weekly BRef batting logs and weekly Statcast aggregates into a
single table with:

* per-week totals (``*_week``)
* year-to-date cumulative counts and rates (``*_ytd``)
* trailing-4-week windowed sums (``trail4w_*``)
* BBE-weighted Statcast year-to-date averages
* rest-of-season targets as both counts (``ros_*``) and per-PA rates
  (``ros_*_per_pa``), plus rate-stat ROS (``ros_obp``, ``ros_slg``)

Inputs:
    data/raw/batting_week_{year}.parquet       (from fetch_game_logs.py)
    data/raw/statcast_agg_week_{year}.parquet  (from fetch_statcast_weekly)

Output:
    data/raw/weekly_snapshots_{year}.parquet

Usage
-----
    python -m src.data.build_snapshots --seasons 2023-2025
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.fetch_statcast import _parse_season_tokens, fetch_statcast_weekly

logger = logging.getLogger(__name__)

# Counting stats we propagate week→ytd→ros→trail4w.
_WEEK_COUNTS: tuple[str, ...] = (
    "pa", "ab", "h", "singles", "doubles", "triples", "hr",
    "r", "rbi", "bb", "ibb", "so", "hbp", "sf", "sh", "sb", "cs",
)

# Statcast columns that are BBE-weighted (mean or rate per BBE).
_STATCAST_RATE_COLS: tuple[str, ...] = (
    "avg_exit_velocity", "avg_launch_angle",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "barrel_rate", "hard_hit_rate", "sweet_spot_rate",
)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division, returning NaN where the denominator is 0 or NaN."""
    den = den.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num.astype(float) / den.where(den > 0, np.nan)
    return out


def _derive_singles(df: pd.DataFrame) -> pd.DataFrame:
    """Derive singles = h - doubles - triples - hr (BRef doesn't supply 1B)."""
    required = {"h", "doubles", "triples", "hr"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise KeyError(f"Cannot derive singles; missing BRef columns: {missing}")
    out = df.copy()
    out["singles"] = (
        out["h"].fillna(0) - out["doubles"].fillna(0)
        - out["triples"].fillna(0) - out["hr"].fillna(0)
    ).clip(lower=0).astype(int)
    return out


def _merge_weekly_sources(
    batting_wk: pd.DataFrame,
    statcast_wk: pd.DataFrame,
) -> pd.DataFrame:
    """Merge weekly BRef batting logs with weekly Statcast aggregates."""
    if "mlbam_id" not in batting_wk.columns:
        raise KeyError("batting_wk requires an 'mlbam_id' column")
    if "mlbam_id" not in statcast_wk.columns:
        raise KeyError("statcast_wk requires an 'mlbam_id' column")

    # Drop rows missing the join key (BRef occasionally returns NA mlbIDs)
    batting_wk = batting_wk.dropna(subset=["mlbam_id"]).copy()
    batting_wk["mlbam_id"] = batting_wk["mlbam_id"].astype("int64")

    statcast_wk = statcast_wk.copy()
    statcast_wk["mlbam_id"] = statcast_wk["mlbam_id"].astype("int64")

    # Statcast duplicates some columns the BRef side already has (season).
    drop_stat = [c for c in ("season",) if c in statcast_wk.columns]
    statcast_wk = statcast_wk.drop(columns=drop_stat)

    merged = batting_wk.merge(
        statcast_wk,
        on=["mlbam_id", "iso_year", "iso_week"],
        how="left",
        validate="one_to_one",
    )
    return merged


def _add_week_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw count columns to `_week` suffix; leave Statcast cols as-is."""
    out = df.copy()
    rename = {c: f"{c}_week" for c in _WEEK_COUNTS if c in out.columns}
    out = out.rename(columns=rename)
    return out


def _add_ytd_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Add year-to-date cumulative counts and BBE-weighted Statcast averages.

    Assumes rows are sorted by ``(mlbam_id, iso_year, iso_week)`` within each
    ``(mlbam_id, season)``. Weekly rows need not be contiguous — off-weeks
    (IL, bench, DNP) produce no row and the next active week's ytd reflects
    only active-week totals (correct behaviour for rate-based modeling).
    """
    out = df.sort_values(
        ["mlbam_id", "season", "iso_year", "iso_week"],
    ).copy()
    grouped = out.groupby(["mlbam_id", "season"])

    # Count-stat cumsum: pa_week → pa_ytd, hr_week → hr_ytd, etc.
    for stat in _WEEK_COUNTS:
        wk_col = f"{stat}_week"
        if wk_col in out.columns:
            out[f"{stat}_ytd"] = grouped[wk_col].cumsum()

    # BBE-weighted Statcast cumulative: for each rate col r with weight w=bbe_count_week,
    # ytd(r) = Σ(w · r) / Σ(w)
    if "bbe_count" in out.columns:
        out = out.rename(columns={"bbe_count": "bbe_count_week"})
    if "bbe_count_week" in out.columns:
        bbe_wk = out["bbe_count_week"].fillna(0)
        out["bbe_count_ytd"] = grouped["bbe_count_week"].cumsum()

        for col in _STATCAST_RATE_COLS:
            if col not in out.columns:
                continue
            weighted = (bbe_wk * out[col].fillna(0))
            num_ytd = weighted.groupby(
                [out["mlbam_id"], out["season"]],
            ).cumsum()
            out[f"{col}_ytd"] = _safe_div(num_ytd, out["bbe_count_ytd"])

        # Rename per-week Statcast rate cols for clarity
        out = out.rename(columns={
            c: f"{c}_week" for c in _STATCAST_RATE_COLS if c in out.columns
        })

    return out


def _add_ytd_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add ytd rate stats: obp, slg, per-PA decompositions, iso, bb_rate, k_rate."""
    out = df.copy()
    pa = out.get("pa_ytd", pd.Series(index=out.index, dtype=float))
    ab = out.get("ab_ytd", pd.Series(index=out.index, dtype=float))
    h = out.get("h_ytd", pd.Series(index=out.index, dtype=float))
    bb = out.get("bb_ytd", pd.Series(index=out.index, dtype=float))
    hbp = out.get("hbp_ytd", pd.Series(index=out.index, dtype=float))
    sf = out.get("sf_ytd", pd.Series(index=out.index, dtype=float))
    singles = out.get("singles_ytd", pd.Series(index=out.index, dtype=float))
    doubles = out.get("doubles_ytd", pd.Series(index=out.index, dtype=float))
    triples = out.get("triples_ytd", pd.Series(index=out.index, dtype=float))
    hr = out.get("hr_ytd", pd.Series(index=out.index, dtype=float))
    so = out.get("so_ytd", pd.Series(index=out.index, dtype=float))

    obp_den = ab.fillna(0) + bb.fillna(0) + hbp.fillna(0) + sf.fillna(0)
    out["obp_ytd"] = _safe_div(h.fillna(0) + bb.fillna(0) + hbp.fillna(0), obp_den)

    total_bases = (
        singles.fillna(0) + 2 * doubles.fillna(0)
        + 3 * triples.fillna(0) + 4 * hr.fillna(0)
    )
    out["slg_ytd"] = _safe_div(total_bases, ab)

    # Per-PA count-stat rate decompositions
    for stat in ("hr", "r", "rbi", "sb", "cs"):
        col = f"{stat}_ytd"
        if col in out.columns:
            out[f"{stat}_per_pa_ytd"] = _safe_div(out[col], pa)

    out["iso_ytd"] = out["slg_ytd"] - _safe_div(h, ab)  # AVG = H/AB
    out["bb_rate_ytd"] = _safe_div(bb, pa)
    out["k_rate_ytd"] = _safe_div(so, pa)
    out["hbp_rate_ytd"] = _safe_div(hbp, pa)
    out["sb_attempt_rate_ytd"] = _safe_div(
        out.get("sb_ytd", 0) + out.get("cs_ytd", 0), pa,
    )
    return out


def _add_trailing_windows(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Add trailing-N-week rolling sums of count stats (recency signal)."""
    out = df.sort_values(
        ["mlbam_id", "season", "iso_year", "iso_week"],
    ).copy()
    grouped = out.groupby(["mlbam_id", "season"], group_keys=False)

    for stat in _WEEK_COUNTS:
        wk = f"{stat}_week"
        if wk in out.columns:
            out[f"trail{window}w_{stat}"] = grouped[wk].apply(
                lambda s: s.rolling(window, min_periods=1).sum(),
            )

    if "bbe_count_week" in out.columns:
        out[f"trail{window}w_bbe_count"] = grouped["bbe_count_week"].apply(
            lambda s: s.rolling(window, min_periods=1).sum(),
        )

    return out


def _add_ros_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest-of-season targets.

    ROS counts: ros_{stat} = season_total_{stat} - ytd_{stat}
    ROS rates: ros_obp, ros_slg computed from ROS counts (formulae preserved
    across shortened seasons because they are PA/AB normalized).
    ROS per-PA rates: ros_hr_per_pa = ros_hr / ros_pa, etc.
    """
    out = df.copy()
    grouped = out.groupby(["mlbam_id", "season"])

    for stat in _WEEK_COUNTS:
        wk = f"{stat}_week"
        ytd = f"{stat}_ytd"
        if wk in out.columns and ytd in out.columns:
            season_total = grouped[wk].transform("sum")
            out[f"ros_{stat}"] = season_total - out[ytd]

    # ROS rate stats (same formulae as ytd but over ROS counts)
    ros_pa = out.get("ros_pa")
    ros_ab = out.get("ros_ab")
    if ros_pa is not None and ros_ab is not None:
        ros_h = out["ros_h"]
        ros_bb = out["ros_bb"]
        ros_hbp = out["ros_hbp"]
        ros_sf = out["ros_sf"]
        obp_den = ros_ab.fillna(0) + ros_bb.fillna(0) + ros_hbp.fillna(0) + ros_sf.fillna(0)
        out["ros_obp"] = _safe_div(
            ros_h.fillna(0) + ros_bb.fillna(0) + ros_hbp.fillna(0),
            obp_den,
        )

        ros_singles = out["ros_singles"]
        ros_doubles = out["ros_doubles"]
        ros_triples = out["ros_triples"]
        ros_hr = out["ros_hr"]
        tb = (
            ros_singles.fillna(0) + 2 * ros_doubles.fillna(0)
            + 3 * ros_triples.fillna(0) + 4 * ros_hr.fillna(0)
        )
        out["ros_slg"] = _safe_div(tb, ros_ab)

        for stat in ("hr", "r", "rbi", "sb"):
            col = f"ros_{stat}"
            if col in out.columns:
                out[f"ros_{stat}_per_pa"] = _safe_div(out[col], ros_pa)

    return out


def build_weekly_snapshots(
    year: int,
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path | None = None,
    min_ytd_pa: int = 50,
    preseason_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build per-(player, ISO-week) snapshots for one season.

    Parameters
    ----------
    year:
        Season year.
    raw_dir:
        Directory containing ``batting_week_{year}.parquet`` and
        ``statcast_agg_week_{year}.parquet``.
    out_dir:
        If given, write ``weekly_snapshots_{year}.parquet`` here.
    min_ytd_pa:
        Minimum ytd plate appearances for inclusion. Drops early-season /
        bench weeks where rate stats are too noisy to model.
    preseason_features:
        Optional DataFrame keyed on ``(mlbam_id, season)`` with preseason
        prior columns. Left-joined onto the snapshots if supplied.

    Returns
    -------
    pd.DataFrame
        One row per ``(mlbam_id, season, iso_year, iso_week)`` meeting the
        ``min_ytd_pa`` threshold.
    """
    raw_dir = Path(raw_dir)

    batting_path = raw_dir / f"batting_week_{year}.parquet"
    if not batting_path.exists():
        raise FileNotFoundError(
            f"{batting_path} missing. Run: python -m src.data.fetch_game_logs "
            f"--seasons {year}"
        )
    batting_wk = pd.read_parquet(batting_path)
    batting_wk = _derive_singles(batting_wk)

    statcast_path = raw_dir / f"statcast_agg_week_{year}.parquet"
    raw_statcast_path = raw_dir / f"statcast_raw_{year}.parquet"
    if not statcast_path.exists() and raw_statcast_path.exists():
        logger.info(
            "No weekly Statcast aggregate at %s — building from raw",
            statcast_path,
        )
        fetch_statcast_weekly(year, out_dir=raw_dir)

    if statcast_path.exists():
        statcast_wk = pd.read_parquet(statcast_path)
    else:
        logger.warning(
            "No weekly Statcast file at %s and no raw cache — snapshot "
            "will have no quality metrics. Fetch with: "
            "python -m src.data.fetch_raw_statcast --seasons %d", statcast_path, year,
        )
        statcast_wk = pd.DataFrame(
            columns=["mlbam_id", "iso_year", "iso_week"],
        )

    merged = _merge_weekly_sources(batting_wk, statcast_wk)
    with_weeks = _add_week_totals(merged)
    with_ytd = _add_ytd_cumulative(with_weeks)
    with_rates = _add_ytd_rates(with_ytd)
    with_trail = _add_trailing_windows(with_rates, window=4)
    with_ros = _add_ros_targets(with_trail)

    # Drop early-season noise
    before = len(with_ros)
    out = with_ros[with_ros["pa_ytd"].fillna(0) >= min_ytd_pa].copy()
    logger.info(
        "Filtered to pa_ytd >= %d: %d -> %d rows", min_ytd_pa, before, len(out),
    )

    if preseason_features is not None:
        out = out.merge(preseason_features, on=["mlbam_id", "season"], how="left")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"weekly_snapshots_{year}.parquet"
        out.to_parquet(
            out_path, engine="pyarrow", compression="zstd", index=False,
        )
        size_mb = out_path.stat().st_size / 1_048_576
        logger.info("Saved -> %s  (%.1f MB, %d rows)", out_path, size_mb, len(out))

    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Build per-(player, ISO-week) snapshots for one or more seasons.",
    )
    parser.add_argument(
        "--seasons", nargs="+", required=True, metavar="YEAR",
        help="Season(s) to build: individual years or a range (e.g. 2023-2025)",
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--min-ytd-pa", type=int, default=50)
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to build: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            build_weekly_snapshots(
                year,
                raw_dir=args.raw_dir,
                out_dir=args.out_dir,
                min_ytd_pa=args.min_ytd_pa,
            )
        except Exception:
            logger.exception("Failed to build snapshots for %d", year)
            failed.append(year)

    if failed:
        logger.error("Failed seasons: %s", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

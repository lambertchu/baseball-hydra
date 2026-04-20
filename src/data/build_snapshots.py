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

_GROUP_KEYS = ["mlbam_id", "season"]
_SORT_KEYS = ["mlbam_id", "season", "iso_year", "iso_week"]

_WEEK_COUNTS: tuple[str, ...] = (
    "pa", "ab", "h", "singles", "doubles", "triples", "hr",
    "r", "rbi", "bb", "ibb", "so", "hbp", "sf", "sh", "sb", "cs",
)

_STATCAST_RATE_COLS: tuple[str, ...] = (
    "avg_exit_velocity", "avg_launch_angle",
    "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle",
    "estimated_slg_using_speedangle",
    "barrel_rate", "hard_hit_rate", "sweet_spot_rate",
)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division, returning NaN where the denominator is 0 or NaN."""
    n = num.to_numpy(dtype=float, copy=False)
    d = den.to_numpy(dtype=float, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(d > 0, n / d, np.nan)
    return pd.Series(result, index=num.index)


def _derive_singles(df: pd.DataFrame) -> pd.DataFrame:
    """Derive singles = h - doubles - triples - hr (BRef doesn't supply 1B)."""
    required = {"h", "doubles", "triples", "hr"}
    if not required.issubset(df.columns):
        raise KeyError(f"Cannot derive singles; missing columns: {required - set(df.columns)}")
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
    if "mlbam_id" not in batting_wk.columns or "mlbam_id" not in statcast_wk.columns:
        raise KeyError("Both inputs require an 'mlbam_id' column")

    # BRef occasionally returns NA mlbIDs for minor-league players already filtered.
    batting_wk = batting_wk.dropna(subset=["mlbam_id"]).copy()
    batting_wk["mlbam_id"] = batting_wk["mlbam_id"].astype("int64")

    statcast_wk = statcast_wk.copy()
    statcast_wk["mlbam_id"] = statcast_wk["mlbam_id"].astype("int64")
    # Drop columns the batting side already provides (BRef values are canonical).
    # Statcast's `week_start_date` is the first-game date in the week, not Monday.
    redundant = ("season", "week_start_date", "week_end_date")
    statcast_wk = statcast_wk.drop(columns=[c for c in redundant if c in statcast_wk.columns])

    return batting_wk.merge(
        statcast_wk,
        on=["mlbam_id", "iso_year", "iso_week"],
        how="left",
        validate="one_to_one",
    )


def _add_week_suffix(df: pd.DataFrame) -> pd.DataFrame:
    """Rename all per-week raw columns to a ``_week`` suffix in one pass.

    Covers counting stats, BBE count, and Statcast rate columns so downstream
    helpers see a single consistent schema and never need a mid-pipeline rename.
    """
    cols = (*_WEEK_COUNTS, "bbe_count", *_STATCAST_RATE_COLS)
    rename = {c: f"{c}_week" for c in cols if c in df.columns}
    return df.rename(columns=rename)


def _apply_count_ytd_trail_ros(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Add ytd cumulative, trailing-window, and ROS count columns in one pass.

    Shares a single sort and groupby across all three transforms to avoid
    O(n log n) resorts and repeated grouper construction. Count-stat
    cumsum/rolling/transform are batched across all columns.
    """
    df = df.sort_values(_SORT_KEYS).reset_index(drop=True)
    grouped = df.groupby(_GROUP_KEYS, sort=False)

    count_cols = [f"{s}_week" for s in _WEEK_COUNTS if f"{s}_week" in df.columns]
    if count_cols:
        cumsums = grouped[count_cols].cumsum()
        totals = grouped[count_cols].transform("sum")
        rolls = (
            grouped[count_cols].rolling(window, min_periods=1).sum()
            .reset_index(level=_GROUP_KEYS, drop=True)
        )
        for stat in _WEEK_COUNTS:
            wk = f"{stat}_week"
            if wk not in count_cols:
                continue
            df[f"{stat}_ytd"] = cumsums[wk]
            df[f"trail{window}w_{stat}"] = rolls[wk]
            df[f"ros_{stat}"] = totals[wk] - cumsums[wk]

    if "bbe_count_week" in df.columns:
        df["bbe_count_ytd"] = grouped["bbe_count_week"].cumsum()
        df[f"trail{window}w_bbe_count"] = (
            grouped["bbe_count_week"].rolling(window, min_periods=1).sum()
            .reset_index(level=_GROUP_KEYS, drop=True)
        )

        # BBE-weighted ytd rate: Σ(bbe_w · r_w) / Σ(bbe_w where r_w is non-null).
        # Using a per-column denominator avoids deflating the average when a
        # week has valid BBEs but a NaN rate (common for xStats in old seasons).
        bbe_wk = df["bbe_count_week"].fillna(0)
        tmp_cols: list[str] = []
        for col in _STATCAST_RATE_COLS:
            wk = f"{col}_week"
            if wk not in df.columns:
                continue
            df[f"__w_{col}"] = bbe_wk * df[wk].fillna(0)
            df[f"__v_{col}"] = bbe_wk.where(df[wk].notna(), 0)
            tmp_cols.extend([f"__w_{col}", f"__v_{col}"])
        if tmp_cols:
            cum = grouped[tmp_cols].cumsum()
            for col in _STATCAST_RATE_COLS:
                wcol, vcol = f"__w_{col}", f"__v_{col}"
                if wcol in cum.columns:
                    df[f"{col}_ytd"] = _safe_div(cum[wcol], cum[vcol])
            df = df.drop(columns=tmp_cols)

    return df


def _obp_slg(
    h: pd.Series, bb: pd.Series, hbp: pd.Series, sf: pd.Series,
    singles: pd.Series, doubles: pd.Series, triples: pd.Series, hr: pd.Series,
    ab: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return (OBP, SLG) from count components. Components may contain NaN."""
    obp_den = ab.fillna(0) + bb.fillna(0) + hbp.fillna(0) + sf.fillna(0)
    obp = _safe_div(h.fillna(0) + bb.fillna(0) + hbp.fillna(0), obp_den)
    tb = singles.fillna(0) + 2 * doubles.fillna(0) + 3 * triples.fillna(0) + 4 * hr.fillna(0)
    slg = _safe_div(tb, ab)
    return obp, slg


def _add_ytd_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ytd rate stats (OBP, SLG, per-PA rates, ISO, discipline)."""
    def col(stat: str) -> pd.Series:
        return df.get(f"{stat}_ytd", pd.Series(index=df.index, dtype=float))

    pa, ab, h, so = col("pa"), col("ab"), col("h"), col("so")
    df["obp_ytd"], df["slg_ytd"] = _obp_slg(
        h, col("bb"), col("hbp"), col("sf"),
        col("singles"), col("doubles"), col("triples"), col("hr"), ab,
    )
    for stat in ("hr", "r", "rbi", "sb", "cs"):
        if f"{stat}_ytd" in df.columns:
            df[f"{stat}_per_pa_ytd"] = _safe_div(col(stat), pa)
    df["iso_ytd"] = df["slg_ytd"] - _safe_div(h, ab)
    df["bb_rate_ytd"] = _safe_div(col("bb"), pa)
    df["k_rate_ytd"] = _safe_div(so, pa)
    df["hbp_rate_ytd"] = _safe_div(col("hbp"), pa)
    df["sb_attempt_rate_ytd"] = _safe_div(col("sb") + col("cs"), pa)
    return df


def _add_ros_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived ROS rate stats. Must be called after ROS count columns exist."""
    if "ros_pa" not in df.columns or "ros_ab" not in df.columns:
        return df

    def col(stat: str) -> pd.Series:
        return df.get(f"ros_{stat}", pd.Series(index=df.index, dtype=float))

    df["ros_obp"], df["ros_slg"] = _obp_slg(
        col("h"), col("bb"), col("hbp"), col("sf"),
        col("singles"), col("doubles"), col("triples"), col("hr"), col("ab"),
    )
    for stat in ("hr", "r", "rbi", "sb"):
        if f"ros_{stat}" in df.columns:
            df[f"ros_{stat}_per_pa"] = _safe_div(col(stat), col("pa"))
    return df


def build_weekly_snapshots(
    year: int,
    raw_dir: str | Path = "data/raw",
    out_dir: str | Path | None = None,
    min_ytd_pa: int = 50,
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
    """
    raw_dir = Path(raw_dir)

    batting_path = raw_dir / f"batting_week_{year}.parquet"
    if not batting_path.exists():
        raise FileNotFoundError(
            f"{batting_path} missing. Run: python -m src.data.fetch_game_logs "
            f"--seasons {year}"
        )
    batting_wk = _derive_singles(pd.read_parquet(batting_path))

    statcast_path = raw_dir / f"statcast_agg_week_{year}.parquet"
    raw_statcast_path = raw_dir / f"statcast_raw_{year}.parquet"
    if not statcast_path.exists() and raw_statcast_path.exists():
        logger.info("No weekly Statcast aggregate at %s — building from raw", statcast_path)
        fetch_statcast_weekly(year, out_dir=raw_dir)

    if statcast_path.exists():
        statcast_wk = pd.read_parquet(statcast_path)
    else:
        logger.warning(
            "No weekly Statcast file at %s and no raw cache — snapshot "
            "will have no quality metrics. Fetch with: "
            "python -m src.data.fetch_raw_statcast --seasons %d", statcast_path, year,
        )
        statcast_wk = pd.DataFrame(columns=["mlbam_id", "iso_year", "iso_week"])

    df = _merge_weekly_sources(batting_wk, statcast_wk)
    df = _add_week_suffix(df)
    df = _apply_count_ytd_trail_ros(df, window=4)
    df = _add_ytd_rates(df)
    df = _add_ros_rates(df)

    before = len(df)
    out = df[df["pa_ytd"].fillna(0) >= min_ytd_pa].copy()
    logger.info("Filtered to pa_ytd >= %d: %d -> %d rows", min_ytd_pa, before, len(out))

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"weekly_snapshots_{year}.parquet"
        out.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
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

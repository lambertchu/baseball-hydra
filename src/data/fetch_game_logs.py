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
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.data.fetch_statcast import _SEASON_DATES, _parse_season_tokens
from src.data.rate_helpers import obp_slg as _obp_slg
from src.data.rate_helpers import safe_div as _safe_div

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

_STATCAST_PA_EVENTS: frozenset[str] = frozenset({
    "single",
    "double",
    "triple",
    "home_run",
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "strikeout",
    "strikeout_double_play",
    "field_out",
    "force_out",
    "fielders_choice",
    "fielders_choice_out",
    "grounded_into_double_play",
    "double_play",
    "field_error",
    "sac_bunt",
    "sac_fly",
    "sac_fly_double_play",
    "catcher_interf",
})
_HIT_EVENTS: frozenset[str] = frozenset({"single", "double", "triple", "home_run"})
_AB_EXCLUDED_EVENTS: frozenset[str] = frozenset({
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_bunt",
    "sac_fly",
    "sac_fly_double_play",
    "catcher_interf",
})
_WEEKLY_COUNT_COLS: tuple[str, ...] = (
    "pa",
    "ab",
    "r",
    "h",
    "doubles",
    "triples",
    "hr",
    "rbi",
    "bb",
    "ibb",
    "so",
    "hbp",
    "sh",
    "sf",
    "gdp",
    "sb",
    "cs",
)
_STEALS_2B_RE = re.compile(r"steals(?: \(\d+\))? 2nd base")
_STEALS_3B_RE = re.compile(r"steals(?: \(\d+\))? 3rd base")
_STEALS_HOME_RE = re.compile(r"steals(?: \(\d+\))? home")


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

    # Drop minor-league rows when present. BRef and pybaseball have used both
    # "Maj-AL/Maj-NL" and "MLB-AL/MLB-NL" as the MLB level label over time.
    if "level" in out.columns:
        lv = out["level"].astype(str)
        out = out[lv.str.startswith(("Maj", "MLB"))].copy()

    for col in _NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "mlbam_id" in out.columns:
        out["mlbam_id"] = pd.to_numeric(out["mlbam_id"], errors="coerce").astype("Int64")

    return out


def _iso_week_bounds(iso_year: int, iso_week: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date.fromisocalendar(int(iso_year), int(iso_week), 1))
    return start, start + pd.Timedelta(days=6)


def _batting_team(row: pd.Series) -> str | None:
    topbot = row.get("inning_topbot")
    if topbot == "Top":
        return row.get("away_team")
    if topbot == "Bot":
        return row.get("home_team")
    return None


def _score_int(row: pd.Series, col: str) -> int:
    value = row.get(col, 0)
    if pd.isna(value):
        return 0
    return int(value)


def _append_runner_event(
    rows: list[tuple[int, int, int, str]],
    kind: str,
    runner_id: object,
    iso_year: int,
    iso_week: int,
) -> None:
    if pd.notna(runner_id):
        rows.append((int(runner_id), int(iso_year), int(iso_week), kind))


def _running_events_from_statcast(raw: pd.DataFrame) -> pd.DataFrame:
    """Infer weekly SB/CS from Statcast descriptions when runner IDs are present."""
    rows: list[tuple[int, int, int, str]] = []
    for _, row in raw.iterrows():
        des = str(row.get("des", ""))
        if not des:
            continue
        iso_year = int(row["iso_year"])
        iso_week = int(row["iso_week"])
        if "steals" in des:
            if _STEALS_2B_RE.search(des):
                _append_runner_event(rows, "sb", row.get("on_1b"), iso_year, iso_week)
            if _STEALS_3B_RE.search(des):
                _append_runner_event(rows, "sb", row.get("on_2b"), iso_year, iso_week)
            if _STEALS_HOME_RE.search(des):
                _append_runner_event(rows, "sb", row.get("on_3b"), iso_year, iso_week)
        if "caught stealing" in des:
            if "caught stealing 2nd" in des:
                _append_runner_event(rows, "cs", row.get("on_1b"), iso_year, iso_week)
            if "caught stealing 3rd" in des:
                _append_runner_event(rows, "cs", row.get("on_2b"), iso_year, iso_week)
            if "caught stealing home" in des:
                _append_runner_event(rows, "cs", row.get("on_3b"), iso_year, iso_week)

    if not rows:
        return pd.DataFrame(columns=["mlbam_id", "iso_year", "iso_week", "sb", "cs"])
    events = pd.DataFrame(rows, columns=["mlbam_id", "iso_year", "iso_week", "kind"])
    out = (
        events.pivot_table(
            index=["mlbam_id", "iso_year", "iso_week"],
            columns="kind",
            values="kind",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in ("sb", "cs"):
        if col not in out.columns:
            out[col] = 0
    return out[["mlbam_id", "iso_year", "iso_week", "sb", "cs"]]


def _runs_from_statcast(pa_rows: pd.DataFrame) -> pd.DataFrame:
    """Attribute runs scored to batter/runner IDs from pre-PA base state."""
    rows: list[tuple[int, int, int, int]] = []
    for _, row in pa_rows.iterrows():
        runs = max(0, _score_int(row, "post_bat_score") - _score_int(row, "bat_score"))
        if runs <= 0:
            continue
        if row.get("events") == "home_run":
            candidates = [
                row.get("batter"),
                row.get("on_3b"),
                row.get("on_2b"),
                row.get("on_1b"),
            ]
        else:
            candidates = [row.get("on_3b"), row.get("on_2b"), row.get("on_1b")]
        scorers = [int(v) for v in candidates if pd.notna(v)]
        for scorer in scorers[:runs]:
            rows.append((scorer, int(row["iso_year"]), int(row["iso_week"]), 1))

    if not rows:
        return pd.DataFrame(columns=["mlbam_id", "iso_year", "iso_week", "r"])
    return (
        pd.DataFrame(rows, columns=["mlbam_id", "iso_year", "iso_week", "r"])
        .groupby(["mlbam_id", "iso_year", "iso_week"], as_index=False)["r"]
        .sum()
    )


def _aggregate_statcast_batting_weekly(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build BRef-like weekly batting counts from full Statcast pitch events."""
    required = {"batter", "game_date", "events", "game_type"}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Statcast batting fallback missing columns: {sorted(missing)}")

    work = raw.loc[raw["game_type"] == "R"].copy()
    if work.empty:
        return pd.DataFrame(columns=["mlbam_id", "iso_year", "iso_week", "season"])
    work["game_date"] = pd.to_datetime(work["game_date"])
    iso = work["game_date"].dt.isocalendar()
    work["iso_year"] = iso["year"].astype(int)
    work["iso_week"] = iso["week"].astype(int)

    pa_rows = work.loc[work["events"].isin(_STATCAST_PA_EVENTS)].copy()
    if {"game_pk", "at_bat_number"}.issubset(pa_rows.columns):
        pa_rows = (
            pa_rows.sort_values(["game_pk", "at_bat_number", "pitch_number"])
            .drop_duplicates(["game_pk", "at_bat_number"], keep="last")
            .copy()
        )
    events = pa_rows["events"].astype(str)
    pa_rows["mlbam_id"] = pa_rows["batter"].astype("int64")
    pa_rows["pa"] = 1
    pa_rows["ab"] = (~events.isin(_AB_EXCLUDED_EVENTS)).astype(int)
    pa_rows["h"] = events.isin(_HIT_EVENTS).astype(int)
    pa_rows["doubles"] = (events == "double").astype(int)
    pa_rows["triples"] = (events == "triple").astype(int)
    pa_rows["hr"] = (events == "home_run").astype(int)
    pa_rows["bb"] = events.isin({"walk", "intent_walk"}).astype(int)
    pa_rows["ibb"] = (events == "intent_walk").astype(int)
    pa_rows["so"] = events.isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa_rows["hbp"] = (events == "hit_by_pitch").astype(int)
    pa_rows["sh"] = (events == "sac_bunt").astype(int)
    pa_rows["sf"] = events.isin({"sac_fly", "sac_fly_double_play"}).astype(int)
    pa_rows["gdp"] = (events == "grounded_into_double_play").astype(int)
    pa_rows["rbi"] = (
        pd.to_numeric(pa_rows.get("post_bat_score"), errors="coerce").fillna(0)
        - pd.to_numeric(pa_rows.get("bat_score"), errors="coerce").fillna(0)
    ).clip(lower=0).astype(int)
    if {"home_team", "away_team", "inning_topbot"}.issubset(pa_rows.columns):
        pa_rows["team"] = pa_rows.apply(_batting_team, axis=1)
    else:
        pa_rows["team"] = None

    group_cols = ["mlbam_id", "iso_year", "iso_week"]
    agg_map = {col: "sum" for col in _WEEKLY_COUNT_COLS if col not in {"r", "sb", "cs"}}
    weekly = pa_rows.groupby(group_cols, as_index=False).agg(agg_map)
    weekly["g"] = pa_rows.groupby(group_cols)["game_pk"].nunique().to_numpy()
    weekly["team"] = pa_rows.groupby(group_cols)["team"].first().to_numpy()

    runs = _runs_from_statcast(pa_rows)
    weekly = weekly.merge(runs, on=group_cols, how="left")
    running = _running_events_from_statcast(work)
    weekly = weekly.merge(running, on=group_cols, how="left")
    for col in ("r", "sb", "cs"):
        weekly[col] = weekly[col].fillna(0).astype(int)

    weekly["singles"] = (
        weekly["h"] - weekly["doubles"] - weekly["triples"] - weekly["hr"]
    ).clip(lower=0)
    weekly["avg"] = _safe_div(weekly["h"], weekly["ab"])
    weekly["obp"], weekly["slg"] = _obp_slg(
        weekly["h"],
        weekly["bb"],
        weekly["hbp"],
        weekly["sf"],
        weekly["singles"],
        weekly["doubles"],
        weekly["triples"],
        weekly["hr"],
        weekly["ab"],
    )
    weekly["ops"] = weekly["obp"] + weekly["slg"]
    weekly["season"] = int(season)
    bounds = weekly[["iso_year", "iso_week"]].apply(
        lambda row: _iso_week_bounds(row["iso_year"], row["iso_week"]),
        axis=1,
    )
    weekly["week_start_date"] = [b[0] for b in bounds]
    weekly["week_end_date"] = [b[1] for b in bounds]
    weekly["level"] = "Statcast"
    weekly["#days"] = pd.NA
    weekly["age"] = pd.NA
    weekly["name"] = weekly["mlbam_id"].map(lambda v: f"Player_{int(v)}")

    columns = [
        "name",
        "age",
        "#days",
        "level",
        "team",
        "g",
        "pa",
        "ab",
        "r",
        "h",
        "doubles",
        "triples",
        "hr",
        "rbi",
        "bb",
        "ibb",
        "so",
        "hbp",
        "sh",
        "sf",
        "gdp",
        "sb",
        "cs",
        "avg",
        "obp",
        "slg",
        "ops",
        "mlbam_id",
        "iso_year",
        "iso_week",
        "week_start_date",
        "week_end_date",
        "season",
    ]
    return weekly[[c for c in columns if c in weekly.columns]]


def _allocate_integer_total(total: int, proxy: pd.Series) -> pd.Series:
    total = int(round(float(total)))
    if total <= 0 or len(proxy) == 0:
        return pd.Series([0] * len(proxy), index=proxy.index, dtype=int)
    weights = pd.to_numeric(proxy, errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(weights.sum()) <= 0.0:
        weights = pd.Series(1.0, index=proxy.index)
    raw = weights * total / float(weights.sum())
    base = raw.apply(lambda v: int(v)).astype(int)
    remainder = total - int(base.sum())
    if remainder > 0:
        frac_order = (raw - base).sort_values(ascending=False).index[:remainder]
        base.loc[frac_order] += 1
    return base.astype(int)


def _overlay_scaled_season_totals(
    weekly: pd.DataFrame,
    season_totals: pd.DataFrame,
    stats: tuple[str, ...] = ("r", "rbi", "sb", "cs"),
) -> pd.DataFrame:
    """Scale selected weekly estimates so each player sums to season totals."""
    if weekly.empty or season_totals.empty or "mlbam_id" not in season_totals.columns:
        return weekly
    out = weekly.copy()
    totals = season_totals.drop_duplicates("mlbam_id").set_index("mlbam_id")
    for mlbam_id, idx in out.groupby("mlbam_id", sort=False).groups.items():
        if mlbam_id not in totals.index:
            continue
        frame = out.loc[idx]
        for stat in stats:
            if stat not in totals.columns or stat not in out.columns:
                continue
            target = totals.at[mlbam_id, stat]
            if pd.isna(target):
                continue
            proxy = frame[stat]
            if float(pd.to_numeric(proxy, errors="coerce").fillna(0.0).sum()) <= 0:
                proxy = frame.get("pa", pd.Series(1.0, index=frame.index))
            out.loc[idx, stat] = _allocate_integer_total(int(target), proxy).to_numpy()
    return out


def _load_season_batting_totals(year: int, raw_dir: Path) -> pd.DataFrame | None:
    batting_path = raw_dir / f"batting_{year}.parquet"
    if not batting_path.exists():
        return None
    totals = pd.read_parquet(batting_path).copy()
    if "mlbam_id" not in totals.columns and "idfg" in totals.columns:
        id_map_path = raw_dir / "id_map_cache.parquet"
        if id_map_path.exists():
            id_map = pd.read_parquet(id_map_path)
            totals = totals.merge(id_map, on="idfg", how="left")
    if "mlbam_id" not in totals.columns:
        return None
    totals = totals.dropna(subset=["mlbam_id"]).copy()
    totals["mlbam_id"] = totals["mlbam_id"].astype("int64")
    return totals


def fetch_batter_weekly_stats_from_statcast(
    year: int,
    out_dir: str | Path = "data/raw",
    force: bool = False,
    delay: float = 5.0,
    min_pa: int = 1,
    calibrate_season_totals: bool = False,
) -> Path:
    """Fetch full Statcast pitch data and derive weekly batting logs.

    This is a fallback for environments where Baseball Reference's daily
    batting endpoint is unavailable. Core PA/AB/H/2B/3B/HR/BB/SO/HBP/SH/SF
    counts are derived from Statcast PA-ending events; R/RBI/SB/CS are
    attributed from score/base-state descriptions. Optional season-total
    calibration is disabled by default because it lets end-of-season totals
    influence midseason cutoff rows.
    """
    try:
        import pybaseball
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

    start_dt, end_dt = _SEASON_DATES[year]
    logger.info(
        "Fetching full Statcast pitch data for weekly batting %d (%s -> %s) ...",
        year,
        start_dt,
        end_dt,
    )
    raw = pybaseball.statcast(start_dt=start_dt, end_dt=end_dt)
    logger.info("  Downloaded %d pitch rows", len(raw))

    weekly = _aggregate_statcast_batting_weekly(raw, season=year)
    if "pa" in weekly.columns:
        weekly = weekly.loc[weekly["pa"].fillna(0) >= min_pa].copy()

    totals = _load_season_batting_totals(year, out_dir)
    if totals is not None:
        meta_cols = [c for c in ("mlbam_id", "name", "age", "team") if c in totals]
        if meta_cols:
            meta = totals[meta_cols].drop_duplicates("mlbam_id")
            replace_cols = [c for c in meta.columns if c != "mlbam_id" and c in weekly]
            weekly = weekly.drop(columns=replace_cols, errors="ignore").merge(
                meta,
                on="mlbam_id",
                how="left",
            )
    if totals is not None and calibrate_season_totals:
        weekly = _overlay_scaled_season_totals(weekly, totals)

    for col in ("name", "age", "team"):
        if col not in weekly.columns:
            weekly[col] = pd.NA
    weekly["name"] = weekly["name"].fillna(
        weekly["mlbam_id"].map(lambda v: f"Player_{int(v)}")
    )
    columns = [
        "name",
        "age",
        "#days",
        "level",
        "team",
        "g",
        "pa",
        "ab",
        "r",
        "h",
        "doubles",
        "triples",
        "hr",
        "rbi",
        "bb",
        "ibb",
        "so",
        "hbp",
        "sh",
        "sf",
        "gdp",
        "sb",
        "cs",
        "avg",
        "obp",
        "slg",
        "ops",
        "mlbam_id",
        "iso_year",
        "iso_week",
        "week_start_date",
        "week_end_date",
        "season",
    ]
    weekly = weekly[[c for c in columns if c in weekly.columns]]
    weekly.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    logger.info(
        "  Saved -> %s  (%.1f MB, %d rows)", out_path, size_mb, len(weekly)
    )

    if delay > 0:
        time.sleep(delay)

    return out_path


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
    failed: list[tuple[int, int]] = []
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
            failed.append((iso_year, iso_week))
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

    # Any per-week failure would produce an incomplete parquet whose YTD/ROS
    # targets would be silently wrong downstream — fail hard instead.
    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)} week(s) for {year}: {failed}. "
            f"Re-run to retry; existing output (if any) is not written."
        )
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
    parser.add_argument(
        "--source",
        choices=["bref", "statcast"],
        default="bref",
        help="Weekly batting source: Baseball Reference range splits or "
        "Statcast-derived fallback (default: bref).",
    )
    parser.add_argument(
        "--calibrate-season-totals",
        action="store_true",
        help="For --source statcast only, scale R/RBI/SB/CS weekly estimates to "
        "local full-season batting totals. Disabled by default to avoid "
        "end-of-season leakage in midseason cutoff snapshots.",
    )
    args = parser.parse_args()

    years = _parse_season_tokens(args.seasons)
    logger.info("Seasons to fetch: %s", years)

    failed: list[int] = []
    for year in years:
        try:
            if args.source == "statcast":
                fetch_batter_weekly_stats_from_statcast(
                    year,
                    out_dir=args.out_dir,
                    force=args.force,
                    delay=args.delay,
                    min_pa=args.min_pa,
                    calibrate_season_totals=args.calibrate_season_totals,
                )
            else:
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

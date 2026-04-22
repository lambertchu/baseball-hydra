"""In-season features derived from weekly snapshot rows (Phase 2).

Consumed by the Phase 2 ROS MTL (``src/models/mtl_ros/``); the preseason MTL
at ``src/models/mtl/`` never sees these features.

Inputs are rows from ``data/raw/weekly_snapshots_{year}.parquet`` (see
``src.data.build_snapshots``) with columns:

* identity / timing: ``mlbam_id``, ``season``, ``iso_week``, ...
* year-to-date: ``pa_ytd``, ``obp_ytd``, ``slg_ytd``, ``hr_per_pa_ytd``, ...
* trailing-4-week counts: ``trail4w_pa``, ``trail4w_hr``, ``trail4w_ab``, ...

The snapshot builder produces trailing-window sums for count stats only, so
``compute_in_season_features`` derives trail4w rate equivalents on the fly
using the same OBP/SLG formulas that ``build_snapshots._add_ytd_rates`` uses
for season-to-date stats. Both sides share
:func:`src.data.rate_helpers.obp_slg` so the formulas cannot drift. If the
caller has already attached a ``trail4w_<rate>`` column (e.g. a downstream
enrichment step), that value is used verbatim.

All 24 output columns are listed in ``IN_SEASON_FEATURE_NAMES`` and are
registered with ``FeatureGroup.IN_SEASON`` in
``src.features.registry``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.rate_helpers import obp_slg, safe_div

# Design choice: a single league-average full-season PA constant used to
# scale ``pa_ytd`` into a season-progress signal. Keeping it as a constant
# here (rather than pulling a per-player preseason PA projection) avoids
# coupling the in-season feature layer to the preseason projector. Revisit
# if we want player-specific denominators.
EXPECTED_SEASON_PA: int = 650

# Public list of in-season feature names in registry order. Mirrors the
# entries in ``src.features.registry.IN_SEASON_FEATURES`` and is kept
# here for ergonomic import from the feature computation module.
IN_SEASON_FEATURE_NAMES: tuple[str, ...] = (
    # YTD passthroughs (10)
    "pa_ytd",
    "obp_ytd",
    "slg_ytd",
    "hr_per_pa_ytd",
    "r_per_pa_ytd",
    "rbi_per_pa_ytd",
    "sb_per_pa_ytd",
    "iso_ytd",
    "bb_rate_ytd",
    "k_rate_ytd",
    # Trail4w rates (10)
    "trail4w_pa",
    "trail4w_obp",
    "trail4w_slg",
    "trail4w_hr_per_pa",
    "trail4w_r_per_pa",
    "trail4w_rbi_per_pa",
    "trail4w_sb_per_pa",
    "trail4w_iso",
    "trail4w_bb_rate",
    "trail4w_k_rate",
    # Derived timing (2)
    "week_index",
    "pa_fraction",
    # IL stubs (2)
    "days_on_il_ytd",
    "has_il_data",
)

_YTD_PASSTHROUGH: tuple[str, ...] = (
    "pa_ytd",
    "obp_ytd",
    "slg_ytd",
    "hr_per_pa_ytd",
    "r_per_pa_ytd",
    "rbi_per_pa_ytd",
    "sb_per_pa_ytd",
    "iso_ytd",
    "bb_rate_ytd",
    "k_rate_ytd",
)


def _col_or_nan(df: pd.DataFrame, name: str) -> pd.Series:
    """Return ``df[name]`` as float, or an all-NaN Series if the column is missing."""
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _trail4w_obp_slg(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute trail4w OBP and SLG from trail4w count components.

    Delegates the arithmetic to :func:`src.data.rate_helpers.obp_slg` so the
    in-season features and the snapshot builder share a single formula, then
    masks rows where every OBP (or SLG) count component was missing so a
    fully-NaN input produces NaN output instead of 0 (the snapshot builder
    doesn't need that guard because its weekly aggregation always produces
    at least one populated component per row).
    """
    h = _col_or_nan(df, "trail4w_h")
    bb = _col_or_nan(df, "trail4w_bb")
    hbp = _col_or_nan(df, "trail4w_hbp")
    sf = _col_or_nan(df, "trail4w_sf")
    ab = _col_or_nan(df, "trail4w_ab")
    singles = _col_or_nan(df, "trail4w_singles")
    doubles = _col_or_nan(df, "trail4w_doubles")
    triples = _col_or_nan(df, "trail4w_triples")
    hr = _col_or_nan(df, "trail4w_hr")

    obp, slg = obp_slg(h, bb, hbp, sf, singles, doubles, triples, hr, ab)

    # Keep NaN rows where every component was missing — otherwise the
    # ``fillna(0)`` inside ``obp_slg`` would silently produce 0/0 → NaN
    # for OBP but masks real "no data" states from "zero on-base".
    obp_has_data = pd.concat([h, bb, hbp, sf, ab], axis=1).notna().any(axis=1)
    slg_has_data = (
        pd.concat([singles, doubles, triples, hr, ab], axis=1).notna().any(axis=1)
    )
    return obp.where(obp_has_data, np.nan), slg.where(slg_has_data, np.nan)


def _trail4w_rate_or_passthrough(
    df: pd.DataFrame,
    out_name: str,
    num_col: str,
    den_col: str,
) -> pd.Series:
    """Return pre-computed trail4w rate column if present, else derive from counts."""
    if out_name in df.columns:
        return df[out_name].astype(float)
    num = _col_or_nan(df, num_col)
    den = _col_or_nan(df, den_col)
    return safe_div(num, den)


def compute_in_season_features(
    weekly_snapshots_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the 24-column in-season feature matrix from weekly snapshots.

    Parameters
    ----------
    weekly_snapshots_df:
        Rows from ``weekly_snapshots_{year}.parquet``. Must carry (at least)
        ``mlbam_id``, ``season``, ``iso_week`` for the derived timing
        features, plus the ytd passthrough columns. Trailing-4-week count
        columns (``trail4w_pa``, ``trail4w_hr``, ...) are used to derive the
        trail4w rate columns; where they are missing, the corresponding rate
        column becomes NaN rather than raising.

    Returns
    -------
    pd.DataFrame
        A DataFrame with exactly the 24 columns enumerated in
        ``IN_SEASON_FEATURE_NAMES``, in the same row order (and index) as
        the input. Safe to ``pd.concat([preseason_X, in_season_X], axis=1)``
        after alignment.
    """
    src = weekly_snapshots_df
    out = pd.DataFrame(index=src.index)

    # 1. YTD passthroughs — float-cast for consistent dtypes; NaN passes through.
    for name in _YTD_PASSTHROUGH:
        out[name] = _col_or_nan(src, name)

    # 2. Trail4w rates — honour pre-computed columns, otherwise derive.
    # PA passthrough from counts column (with NaN fallback).
    out["trail4w_pa"] = _col_or_nan(src, "trail4w_pa")

    # Compute derived OBP/SLG once; use only the components we need to fill
    # whichever columns weren't pre-computed on the snapshot.
    need_obp = "trail4w_obp" not in src.columns
    need_slg = "trail4w_slg" not in src.columns
    if need_obp or need_slg:
        derived_obp, derived_slg = _trail4w_obp_slg(src)
    out["trail4w_obp"] = (
        src["trail4w_obp"].astype(float) if not need_obp else derived_obp
    )
    out["trail4w_slg"] = (
        src["trail4w_slg"].astype(float) if not need_slg else derived_slg
    )

    # Per-PA rate decomposition (HR, R, RBI, SB) + discipline (BB, K) +
    # ISO = SLG - AVG.  Each honours a pre-computed column when available.
    out["trail4w_hr_per_pa"] = _trail4w_rate_or_passthrough(
        src, "trail4w_hr_per_pa", "trail4w_hr", "trail4w_pa"
    )
    out["trail4w_r_per_pa"] = _trail4w_rate_or_passthrough(
        src, "trail4w_r_per_pa", "trail4w_r", "trail4w_pa"
    )
    out["trail4w_rbi_per_pa"] = _trail4w_rate_or_passthrough(
        src, "trail4w_rbi_per_pa", "trail4w_rbi", "trail4w_pa"
    )
    out["trail4w_sb_per_pa"] = _trail4w_rate_or_passthrough(
        src, "trail4w_sb_per_pa", "trail4w_sb", "trail4w_pa"
    )
    out["trail4w_bb_rate"] = _trail4w_rate_or_passthrough(
        src, "trail4w_bb_rate", "trail4w_bb", "trail4w_pa"
    )
    out["trail4w_k_rate"] = _trail4w_rate_or_passthrough(
        src, "trail4w_k_rate", "trail4w_so", "trail4w_pa"
    )

    if "trail4w_iso" in src.columns:
        out["trail4w_iso"] = src["trail4w_iso"].astype(float)
    else:
        slg_series = out["trail4w_slg"]
        avg_series = safe_div(
            _col_or_nan(src, "trail4w_h"), _col_or_nan(src, "trail4w_ab")
        )
        out["trail4w_iso"] = slg_series - avg_series

    # 3. Derived timing features.
    if {"mlbam_id", "season", "iso_week"}.issubset(src.columns) and len(src) > 0:
        iso_week = src["iso_week"].astype(float)
        min_week = src.groupby(["mlbam_id", "season"])["iso_week"].transform("min")
        out["week_index"] = (iso_week - min_week.astype(float)).astype(float)
    else:
        out["week_index"] = pd.Series(np.nan, index=src.index, dtype=float)

    # pa_fraction = pa_ytd / EXPECTED_SEASON_PA. Constant denominator keeps
    # this feature decoupled from per-player preseason PA projections.
    out["pa_fraction"] = out["pa_ytd"] / float(EXPECTED_SEASON_PA)

    # 4. IL stubs — constant zeros pending a real injured-list feed.
    out["days_on_il_ytd"] = pd.Series(0, index=src.index, dtype="int64")
    out["has_il_data"] = pd.Series(0, index=src.index, dtype="int64")

    # Emit columns in the canonical registry order.
    return out.loc[:, list(IN_SEASON_FEATURE_NAMES)]

"""Weekly sequence features for the Phase 3 GRU ROS model.

The feed-forward Phase 2 model sees one cutoff row at a time.  Phase 3 adds
the within-season path to that cutoff: week-level mechanics, plate-discipline,
and outcome signals derived only from ``*_week`` snapshot columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.rate_helpers import obp_slg, safe_div

EXPECTED_SEASON_PA: int = 650

MECHANICS_SEQUENCE_FEATURES: tuple[str, ...] = (
    "seq_bbe_count",
    "seq_avg_exit_velocity",
    "seq_ev_p95",
    "seq_max_exit_velocity",
    "seq_avg_launch_angle",
    "seq_barrel_rate",
    "seq_hard_hit_rate",
    "seq_sweet_spot_rate",
    "seq_estimated_woba_using_speedangle",
    "seq_estimated_ba_using_speedangle",
    "seq_estimated_slg_using_speedangle",
)

PLATE_SEQUENCE_FEATURES: tuple[str, ...] = (
    "seq_week_pa",
    "seq_week_bb_rate",
    "seq_week_k_rate",
    "seq_week_hbp_rate",
    "seq_week_contact_rate",
    "seq_week_sb_attempt_rate",
)

OUTCOME_SEQUENCE_FEATURES: tuple[str, ...] = (
    "seq_week_obp",
    "seq_week_slg",
    "seq_week_iso",
    "seq_week_hr_per_pa",
    "seq_week_r_per_pa",
    "seq_week_rbi_per_pa",
    "seq_week_sb_per_pa",
    "seq_week_index",
    "seq_pa_ytd",
    "seq_pa_fraction",
)

SEQUENCE_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "mechanics": MECHANICS_SEQUENCE_FEATURES,
    "plate": PLATE_SEQUENCE_FEATURES,
    "outcome": OUTCOME_SEQUENCE_FEATURES,
}

SEQUENCE_FEATURE_NAMES: tuple[str, ...] = (
    *MECHANICS_SEQUENCE_FEATURES,
    *PLATE_SEQUENCE_FEATURES,
    *OUTCOME_SEQUENCE_FEATURES,
)

BLEND_FEATURE_NAMES: tuple[str, str] = ("seq_pa_ytd", "seq_week_index")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return a numeric source column, or NaN when the snapshot lacks it."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(np.nan, index=df.index, dtype=float)


def _copy_week_col(src: pd.DataFrame, out: pd.DataFrame, dst: str, source: str) -> None:
    out[dst] = _col(src, source)


def compute_weekly_sequence_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return Phase 3 week-level sequence features aligned to ``snapshots``.

    Only columns ending in ``_week`` plus current-row timing/YTD PA are used.
    No future rows or ROS targets participate in the calculation.
    """
    src = snapshots
    out = pd.DataFrame(index=src.index)

    _copy_week_col(src, out, "seq_bbe_count", "bbe_count_week")
    _copy_week_col(src, out, "seq_avg_exit_velocity", "avg_exit_velocity_week")
    _copy_week_col(src, out, "seq_ev_p95", "ev_p95_week")
    _copy_week_col(src, out, "seq_max_exit_velocity", "max_exit_velocity_week")
    _copy_week_col(src, out, "seq_avg_launch_angle", "avg_launch_angle_week")
    _copy_week_col(src, out, "seq_barrel_rate", "barrel_rate_week")
    _copy_week_col(src, out, "seq_hard_hit_rate", "hard_hit_rate_week")
    _copy_week_col(src, out, "seq_sweet_spot_rate", "sweet_spot_rate_week")
    _copy_week_col(
        src,
        out,
        "seq_estimated_woba_using_speedangle",
        "estimated_woba_using_speedangle_week",
    )
    _copy_week_col(
        src,
        out,
        "seq_estimated_ba_using_speedangle",
        "estimated_ba_using_speedangle_week",
    )
    _copy_week_col(
        src,
        out,
        "seq_estimated_slg_using_speedangle",
        "estimated_slg_using_speedangle_week",
    )

    pa = _col(src, "pa_week")
    ab = _col(src, "ab_week")
    h = _col(src, "h_week")
    bb = _col(src, "bb_week")
    hbp = _col(src, "hbp_week")
    sf = _col(src, "sf_week")
    so = _col(src, "so_week")
    singles = _col(src, "singles_week")
    doubles = _col(src, "doubles_week")
    triples = _col(src, "triples_week")
    hr = _col(src, "hr_week")
    sb = _col(src, "sb_week")
    cs = _col(src, "cs_week")

    out["seq_week_pa"] = pa
    out["seq_week_obp"], out["seq_week_slg"] = obp_slg(
        h, bb, hbp, sf, singles, doubles, triples, hr, ab
    )
    out["seq_week_iso"] = out["seq_week_slg"] - safe_div(h, ab)
    out["seq_week_hr_per_pa"] = safe_div(hr, pa)
    out["seq_week_r_per_pa"] = safe_div(_col(src, "r_week"), pa)
    out["seq_week_rbi_per_pa"] = safe_div(_col(src, "rbi_week"), pa)
    out["seq_week_sb_per_pa"] = safe_div(sb, pa)
    out["seq_week_bb_rate"] = safe_div(bb, pa)
    out["seq_week_k_rate"] = safe_div(so, pa)
    out["seq_week_hbp_rate"] = safe_div(hbp, pa)
    out["seq_week_contact_rate"] = (
        1.0
        - out["seq_week_bb_rate"]
        - out["seq_week_k_rate"]
        - out["seq_week_hbp_rate"]
    )
    out["seq_week_sb_attempt_rate"] = safe_div(sb + cs, pa)

    if {"mlbam_id", "season", "iso_week"}.issubset(src.columns) and len(src) > 0:
        iso_week = _col(src, "iso_week")
        min_week = src.groupby(["mlbam_id", "season"])["iso_week"].transform("min")
        out["seq_week_index"] = iso_week - min_week.astype(float)
    else:
        out["seq_week_index"] = np.nan

    out["seq_pa_ytd"] = _col(src, "pa_ytd")
    out["seq_pa_fraction"] = out["seq_pa_ytd"] / float(EXPECTED_SEASON_PA)

    return out.loc[:, list(SEQUENCE_FEATURE_NAMES)]

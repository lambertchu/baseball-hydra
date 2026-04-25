"""Tests for Phase 3 ROS sequence feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.ros.features import compute_weekly_sequence_features


def test_weekly_sequence_features_derive_rate_stats() -> None:
    snapshots = pd.DataFrame(
        [
            {
                "mlbam_id": 1,
                "season": 2025,
                "iso_year": 2025,
                "iso_week": 14,
                "pa_week": 10,
                "ab_week": 8,
                "h_week": 3,
                "singles_week": 1,
                "doubles_week": 1,
                "triples_week": 0,
                "hr_week": 1,
                "r_week": 2,
                "rbi_week": 3,
                "bb_week": 1,
                "so_week": 2,
                "hbp_week": 1,
                "sf_week": 0,
                "sb_week": 1,
                "cs_week": 0,
                "pa_ytd": 55,
                "avg_exit_velocity_week": 91.0,
            }
        ]
    )

    feats = compute_weekly_sequence_features(snapshots)

    assert feats.loc[0, "seq_week_obp"] == 0.5  # (H+BB+HBP)/(AB+BB+HBP+SF)
    assert feats.loc[0, "seq_week_slg"] == 0.875  # (1 + 2 + 0 + 4) / AB
    assert feats.loc[0, "seq_week_hr_per_pa"] == 0.1
    assert feats.loc[0, "seq_week_r_per_pa"] == 0.2
    assert feats.loc[0, "seq_week_rbi_per_pa"] == 0.3
    assert feats.loc[0, "seq_week_sb_per_pa"] == 0.1
    assert feats.loc[0, "seq_week_bb_rate"] == 0.1
    assert feats.loc[0, "seq_week_k_rate"] == 0.2
    assert feats.loc[0, "seq_week_iso"] == 0.5
    assert feats.loc[0, "seq_week_index"] == 0.0
    assert feats.loc[0, "seq_pa_fraction"] == 55 / 650
    assert feats.loc[0, "seq_avg_exit_velocity"] == 91.0


def test_weekly_sequence_features_keep_nan_when_denominator_zero() -> None:
    snapshots = pd.DataFrame(
        [
            {
                "mlbam_id": 1,
                "season": 2025,
                "iso_year": 2025,
                "iso_week": 14,
                "pa_week": 0,
                "ab_week": 0,
                "h_week": 0,
                "singles_week": 0,
                "doubles_week": 0,
                "triples_week": 0,
                "hr_week": 0,
                "bb_week": 0,
                "hbp_week": 0,
                "sf_week": 0,
                "so_week": 0,
                "r_week": 0,
                "rbi_week": 0,
                "sb_week": 0,
                "cs_week": 0,
                "pa_ytd": 0,
            }
        ]
    )

    feats = compute_weekly_sequence_features(snapshots)

    assert np.isnan(feats.loc[0, "seq_week_obp"])
    assert np.isnan(feats.loc[0, "seq_week_slg"])
    assert np.isnan(feats.loc[0, "seq_week_hr_per_pa"])

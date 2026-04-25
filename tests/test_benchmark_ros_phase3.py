"""Tests for Phase 3 registration and benchmark prediction plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ros as br  # noqa: E402
import generate_ros_projections as grp  # noqa: E402
from src.eval.ros_metrics import ROS_RATE_TARGETS  # noqa: E402


class _FakePhase3Ensemble:
    def __init__(self) -> None:
        self.is_fitted_ = True

    def predict(self, snapshots: pd.DataFrame, phase2_features: pd.DataFrame) -> dict:
        n = len(snapshots)
        base = np.array([0.32, 0.42, 0.03, 0.12, 0.13, 0.02])
        offsets = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
        q = base[None, :, None] + offsets[None, None, :] + np.zeros((n, 1, 1))
        return {"quantiles": q, "pa_remaining": np.full((n, 1), 250.0)}


def _snapshots() -> pd.DataFrame:
    rows = []
    for week in (14, 15, 16):
        rows.append(
            {
                "mlbam_id": 1,
                "season": 2025,
                "iso_year": 2025,
                "iso_week": week,
                "pa_ytd": 60.0 + week,
                "obp_ytd": 0.330,
                "slg_ytd": 0.430,
                "hr_per_pa_ytd": 0.030,
                "r_per_pa_ytd": 0.120,
                "rbi_per_pa_ytd": 0.130,
                "sb_per_pa_ytd": 0.020,
                "iso_ytd": 0.160,
                "bb_rate_ytd": 0.080,
                "k_rate_ytd": 0.220,
                "trail4w_pa": 40.0,
                "trail4w_h": 10.0,
                "trail4w_bb": 3.0,
                "trail4w_hbp": 0.5,
                "trail4w_sf": 0.5,
                "trail4w_ab": 35.0,
                "trail4w_singles": 6.0,
                "trail4w_doubles": 2.0,
                "trail4w_triples": 0.5,
                "trail4w_hr": 1.0,
                "trail4w_r": 5.0,
                "trail4w_rbi": 5.0,
                "trail4w_sb": 0.5,
                "trail4w_so": 10.0,
                "ros_obp": 0.340,
                "ros_slg": 0.450,
                "ros_hr_per_pa": 0.040,
                "ros_r_per_pa": 0.130,
                "ros_rbi_per_pa": 0.140,
                "ros_sb_per_pa": 0.025,
                "ros_pa": 250.0,
            }
        )
    return pd.DataFrame(rows)


def test_phase3_baseline_registered() -> None:
    assert "phase3" in br.ALL_BASELINES
    assert "phase3" in br._BASELINES_NEED_PRESEASON
    assert "phase3" in br._BASELINES_EMIT_QUANTILES
    assert br._BASELINE_DISPLAY["phase3"] == "Phase3"


def test_train_only_split_override_drops_yaml_holdouts() -> None:
    cfg = {
        "splits": {
            "train_end_season": 2022,
            "val_season": 2023,
            "test_season": 2024,
        }
    }

    br._force_train_only_splits(cfg, 2024)
    assert cfg["splits"] == {"train_end_season": 2024}

    grp._force_train_only_splits(cfg, 2025)
    assert cfg["splits"] == {"train_end_season": 2025}


def test_predict_phase3_maps_full_context_back_to_checkpoint_rows() -> None:
    context = _snapshots()
    checkpoint_rows = context.iloc[[1]].reset_index(drop=True)

    point, q = br.predict_phase3(
        rows=checkpoint_rows,
        yearly_snapshots=context,
        ensemble=_FakePhase3Ensemble(),
        phase3_config={"model": {"taus": [0.05, 0.25, 0.5, 0.75, 0.95]}},
        preseason=None,
    )

    assert list(point.columns) == list(ROS_RATE_TARGETS)
    assert point.shape == (1, 6)
    assert q.shape == (1, 6, 5)
    np.testing.assert_allclose(point.iloc[0].to_numpy(), q[0, :, 2])

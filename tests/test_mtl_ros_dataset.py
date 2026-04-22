"""Tests for ROSSnapshotDataset and compute_sample_weights.

The dataset wraps weekly-snapshot rows as a PyTorch Dataset suitable for
the ROS quantile MTL.  Key contracts:

* min_ytd_pa filter drops early-season rows where rate targets are too noisy
* NaN in features become 0.0 (matching preseason BatterDataset)
* __getitem__ returns a dict with x / y / pa_target / weight
* compute_sample_weights combines recency decay with sqrt(ros_pa+1) and
  mean-normalizes to 1.0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.mtl_ros.dataset import ROSSnapshotDataset, compute_sample_weights


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


_RATE_TARGET_COLS = (
    "ros_obp",
    "ros_slg",
    "ros_hr_per_pa",
    "ros_r_per_pa",
    "ros_rbi_per_pa",
    "ros_sb_per_pa",
)
_FEATURE_COLS = ("pa_ytd", "obp_ytd", "slg_ytd", "hr_per_pa_ytd")


def _make_snapshot_rows(rows: list[dict]) -> pd.DataFrame:
    """Fill in defaults so every row has the columns the dataset expects."""
    frame_rows: list[dict] = []
    for i, r in enumerate(rows):
        base = {
            "mlbam_id": r.get("mlbam_id", 100 + i),
            "season": r.get("season", 2024),
            "iso_year": r.get("iso_year", r.get("season", 2024)),
            "iso_week": r.get("iso_week", 20),
            "pa_ytd": r.get("pa_ytd", 100.0),
            "obp_ytd": r.get("obp_ytd", 0.330),
            "slg_ytd": r.get("slg_ytd", 0.420),
            "hr_per_pa_ytd": r.get("hr_per_pa_ytd", 0.04),
            "ros_pa": r.get("ros_pa", 200.0),
            "ros_obp": r.get("ros_obp", 0.325),
            "ros_slg": r.get("ros_slg", 0.415),
            "ros_hr_per_pa": r.get("ros_hr_per_pa", 0.035),
            "ros_r_per_pa": r.get("ros_r_per_pa", 0.12),
            "ros_rbi_per_pa": r.get("ros_rbi_per_pa", 0.11),
            "ros_sb_per_pa": r.get("ros_sb_per_pa", 0.02),
        }
        # Allow overriding any of the above explicitly.
        base.update(r)
        frame_rows.append(base)
    return pd.DataFrame(frame_rows)


# ---------------------------------------------------------------------------
# ROSSnapshotDataset
# ---------------------------------------------------------------------------


class TestFilterMinPa:
    def test_filter_min_pa(self):
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 10.0},
                {"pa_ytd": 49.9},
                {"pa_ytd": 50.0},
                {"pa_ytd": 200.0},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=50,
        )
        assert len(ds) == 2
        assert (ds.filtered_snapshots["pa_ytd"] >= 50).all()


class TestGetitem:
    def test_getitem_dict_shape(self):
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 100.0, "ros_pa": 220.0},
                {"pa_ytd": 200.0, "ros_pa": 180.0},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=0,
        )
        item = ds[0]
        assert set(item.keys()) == {"x", "y", "pa_target", "weight"}
        assert item["x"].shape == (len(_FEATURE_COLS),)
        assert item["y"].shape == (len(_RATE_TARGET_COLS),)
        assert item["pa_target"].shape == (1,)
        assert item["weight"].shape == ()  # scalar
        assert item["x"].dtype == torch.float32
        assert item["y"].dtype == torch.float32
        assert item["pa_target"].dtype == torch.float32
        assert item["weight"].dtype == torch.float32

    def test_feature_nan_handling(self):
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 150.0, "slg_ytd": np.nan, "hr_per_pa_ytd": np.nan},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=0,
        )
        x = ds[0]["x"].numpy()
        # Features with NaN → 0.0
        slg_idx = _FEATURE_COLS.index("slg_ytd")
        hr_idx = _FEATURE_COLS.index("hr_per_pa_ytd")
        assert x[slg_idx] == 0.0
        assert x[hr_idx] == 0.0
        # Non-NaN feature unchanged
        pa_idx = _FEATURE_COLS.index("pa_ytd")
        assert x[pa_idx] == pytest.approx(150.0)

    def test_len_matches_filtered_rows(self):
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 10.0},
                {"pa_ytd": 60.0},
                {"pa_ytd": 100.0},
                {"pa_ytd": 300.0},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=50,
        )
        assert len(ds) == 3

    def test_pa_target_value(self):
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 100.0, "ros_pa": 245.0},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=0,
        )
        assert ds[0]["pa_target"].item() == pytest.approx(245.0)

    def test_pa_target_nan_clamped_to_zero(self):
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 100.0, "ros_pa": np.nan},
                {"pa_ytd": 120.0, "ros_pa": -5.0},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=0,
        )
        assert ds[0]["pa_target"].item() == 0.0
        assert ds[1]["pa_target"].item() == 0.0

    def test_sample_weight_col_passthrough(self):
        """When sample_weight_col is provided, those values are used verbatim."""
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 100.0, "season": 2020, "ros_pa": 100.0, "my_weight": 0.5},
                {"pa_ytd": 100.0, "season": 2024, "ros_pa": 300.0, "my_weight": 2.5},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            sample_weight_col="my_weight",
            min_ytd_pa=0,
        )
        # Weights are passed through unchanged (not recomputed).
        assert ds[0]["weight"].item() == pytest.approx(0.5)
        assert ds[1]["weight"].item() == pytest.approx(2.5)

    def test_uniform_weights_default(self):
        """Without sample_weight_col, all weights are 1.0."""
        rows = _make_snapshot_rows(
            [
                {"pa_ytd": 100.0},
                {"pa_ytd": 200.0},
            ]
        )
        ds = ROSSnapshotDataset(
            rows,
            feature_cols=list(_FEATURE_COLS),
            rate_target_cols=list(_RATE_TARGET_COLS),
            min_ytd_pa=0,
        )
        assert ds[0]["weight"].item() == pytest.approx(1.0)
        assert ds[1]["weight"].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_sample_weights
# ---------------------------------------------------------------------------


class TestComputeSampleWeights:
    def test_compute_sample_weights_sqrt_ros_pa(self):
        """With constant season, weights scale as sqrt(ros_pa+1) then mean-normalize."""
        rows = _make_snapshot_rows(
            [
                {"season": 2024, "ros_pa": 0.0},
                {"season": 2024, "ros_pa": 99.0},
                {"season": 2024, "ros_pa": 399.0},
            ]
        )
        w = compute_sample_weights(rows, recency_lambda=0.0)
        # Raw scales: sqrt(1)=1, sqrt(100)=10, sqrt(400)=20 → sum=31, mean=31/3
        raw = np.array([1.0, 10.0, 20.0])
        expected = raw / raw.mean()
        np.testing.assert_allclose(w, expected, rtol=1e-5)
        # Mean-normalized to 1.0
        assert w.mean() == pytest.approx(1.0)

    def test_compute_sample_weights_recency_decay(self):
        """Older seasons have strictly smaller pre-normalization weight."""
        rows = _make_snapshot_rows(
            [
                {"season": 2020, "ros_pa": 100.0},
                {"season": 2022, "ros_pa": 100.0},
                {"season": 2024, "ros_pa": 100.0},
            ]
        )
        w = compute_sample_weights(rows, recency_lambda=0.30)
        # All rows have the same ros_pa so differences come purely from recency.
        # The most recent season has the highest weight.
        assert w[2] > w[1] > w[0]
        # Mean normalized to 1.0
        assert w.mean() == pytest.approx(1.0, rel=1e-5)

    def test_compute_sample_weights_zero_ros_pa(self):
        """ros_pa=0 rows still receive a finite positive weight via the +1 guard."""
        rows = _make_snapshot_rows(
            [
                {"season": 2024, "ros_pa": 0.0},
                {"season": 2024, "ros_pa": 0.0},
                {"season": 2024, "ros_pa": 100.0},
            ]
        )
        w = compute_sample_weights(rows, recency_lambda=0.0)
        assert np.all(np.isfinite(w))
        assert (w > 0).all()
        # The two ros_pa=0 rows must share the same weight.
        assert w[0] == pytest.approx(w[1])
        # And be strictly less than the ros_pa=100 row.
        assert w[0] < w[2]

    def test_compute_sample_weights_output_dtype_and_length(self):
        rows = _make_snapshot_rows(
            [
                {"season": 2024, "ros_pa": 100.0},
                {"season": 2023, "ros_pa": 200.0},
            ]
        )
        w = compute_sample_weights(rows, recency_lambda=0.30)
        assert w.dtype == np.float32
        assert w.shape == (len(rows),)

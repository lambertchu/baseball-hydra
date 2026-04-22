"""Tests for compute_sample_weights.

The weight formula combines recency decay with sqrt(ros_pa+1) and
mean-normalizes to 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.mtl_ros.dataset import compute_sample_weights


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_snapshot_rows(rows: list[dict]) -> pd.DataFrame:
    """Fill in defaults so every row has the columns compute_sample_weights expects."""
    frame_rows: list[dict] = []
    for i, r in enumerate(rows):
        base = {
            "mlbam_id": r.get("mlbam_id", 100 + i),
            "season": r.get("season", 2024),
            "ros_pa": r.get("ros_pa", 200.0),
        }
        # Allow overriding any of the above explicitly.
        base.update(r)
        frame_rows.append(base)
    return pd.DataFrame(frame_rows)


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

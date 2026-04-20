"""Tests for ROS evaluation metrics.

Covers:
- Pinball loss definition and quantile consistency
- PIT coverage on calibrated and miscalibrated predictions
- PA checkpoint row selection (first-crossing semantics, missing players)
- Target-name resolution and error handling
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.ros_metrics import (
    DEFAULT_PA_CHECKPOINTS,
    DEFAULT_QUANTILE_LEVELS,
    ROS_RATE_TARGETS,
    ROS_TARGET_DISPLAY,
    ROS_YTD_RATES,
    pa_checkpoint_rows,
    pinball_loss,
    pit_coverage,
    quantile_loss,
)


# ---------------------------------------------------------------------------
# Target-name conventions
# ---------------------------------------------------------------------------

class TestTargetConventions:
    def test_all_tuples_same_length(self):
        assert len(ROS_RATE_TARGETS) == len(ROS_TARGET_DISPLAY) == len(ROS_YTD_RATES) == 6

    def test_ros_target_format(self):
        assert all(name.startswith("ros_") for name in ROS_RATE_TARGETS)

    def test_ytd_target_format(self):
        assert all(name.endswith("_ytd") for name in ROS_YTD_RATES)

    def test_defaults(self):
        assert DEFAULT_PA_CHECKPOINTS == (50, 100, 200, 400)
        assert DEFAULT_QUANTILE_LEVELS == (0.05, 0.25, 0.50, 0.75, 0.95)


# ---------------------------------------------------------------------------
# pinball_loss
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_perfect_prediction_zero_loss(self):
        y = np.array([0.3, 0.4, 0.5])
        assert pinball_loss(y, y, tau=0.5) == 0.0

    def test_symmetric_median(self):
        # At tau=0.5, pinball loss is symmetric: |y - q| / 2.
        y = np.array([0.0, 1.0, 2.0])
        q = np.array([1.0, 1.0, 1.0])
        expected = float(np.mean(np.abs(y - q)) / 2.0)
        assert pinball_loss(y, q, tau=0.5) == pytest.approx(expected)

    def test_asymmetry_below_quantile(self):
        # tau=0.1: under-prediction (y > q) heavily penalized (factor 0.1)
        # vs over-prediction (y < q) less penalized (factor 0.9). Wait, check:
        # diff = y - q. If y > q, diff > 0, loss = tau*diff = 0.1*diff.
        # If y < q, diff < 0, loss = (tau-1)*diff = (-0.9)*(negative) = 0.9*|diff|.
        y_over = np.array([1.0])
        y_under = np.array([-1.0])
        q = np.array([0.0])
        loss_over = pinball_loss(y_over, q, tau=0.1)
        loss_under = pinball_loss(y_under, q, tau=0.1)
        assert loss_over == pytest.approx(0.1)
        assert loss_under == pytest.approx(0.9)

    def test_optimal_quantile_minimizes_loss(self):
        # For a sample from N(0,1), the 0.25 quantile prediction is -0.6745.
        # Any other constant prediction should yield higher pinball loss at tau=0.25.
        rng = np.random.default_rng(0)
        y = rng.standard_normal(5000)
        tau = 0.25
        optimal = np.quantile(y, tau)
        loss_at_optimal = pinball_loss(y, np.full_like(y, optimal), tau)
        for q in (optimal - 0.5, optimal + 0.5, 0.0, 1.0):
            loss = pinball_loss(y, np.full_like(y, q), tau)
            assert loss >= loss_at_optimal - 1e-6


# ---------------------------------------------------------------------------
# quantile_loss
# ---------------------------------------------------------------------------

class TestQuantileLoss:
    def _synthetic(self, n=200, n_targets=2, seed=0):
        rng = np.random.default_rng(seed)
        y_true = rng.normal(0.3, 0.05, size=(n, n_targets))
        return y_true, rng

    def test_output_structure(self):
        y_true, rng = self._synthetic()
        taus = [0.1, 0.5, 0.9]
        # Stack quantile predictions (n, n_targets, n_quantiles)
        y_quant = np.stack(
            [np.quantile(y_true, tau, axis=0) + 0 * y_true for tau in taus],
            axis=-1,
        )
        result = quantile_loss(y_true, y_quant, taus, target_names=["A", "B"])

        assert set(result) == {"taus", "per_target", "aggregate"}
        assert result["taus"] == taus
        assert set(result["per_target"]) == {"A", "B"}
        for target in ("A", "B"):
            assert set(result["per_target"][target]) == {"mean_pinball", "per_tau"}
            assert set(result["per_target"][target]["per_tau"]) == set(taus)
        assert "mean_pinball" in result["aggregate"]

    def test_median_prediction_beats_offset(self):
        # The sample median should yield lower pinball than an offset constant at tau=0.5.
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.normal(0.3, 0.1, size=(n, 1))
        median = float(np.median(y_true))

        q_at_median = np.full((n, 1, 1), median)
        q_offset = np.full((n, 1, 1), median + 0.1)

        loss_optimal = quantile_loss(y_true, q_at_median, [0.5])["aggregate"]["mean_pinball"]
        loss_offset = quantile_loss(y_true, q_offset, [0.5])["aggregate"]["mean_pinball"]
        assert loss_optimal < loss_offset

    def test_shape_mismatch_raises(self):
        y_true = np.zeros((5, 3))
        bad_quant = np.zeros((5, 2, 3))  # wrong n_targets
        with pytest.raises(ValueError, match="incompatible"):
            quantile_loss(y_true, bad_quant, [0.25, 0.5, 0.75])

    def test_taus_mismatch_raises(self):
        y_true = np.zeros((5, 2))
        y_quant = np.zeros((5, 2, 3))
        with pytest.raises(ValueError, match="3 quantiles but 2 taus"):
            quantile_loss(y_true, y_quant, [0.25, 0.75])

    def test_dataframe_target_names(self):
        y_df = pd.DataFrame({"OBP": [0.3, 0.4, 0.5], "SLG": [0.4, 0.5, 0.6]})
        q = np.full((3, 2, 1), 0.45)
        result = quantile_loss(y_df, q, [0.5])
        assert set(result["per_target"]) == {"OBP", "SLG"}


# ---------------------------------------------------------------------------
# pit_coverage
# ---------------------------------------------------------------------------

class TestPitCoverage:
    def test_calibrated_predictions(self):
        # Calibrated quantiles = sample quantiles. Empirical coverage ≈ nominal.
        rng = np.random.default_rng(0)
        n = 10000
        y_true = rng.normal(0.0, 1.0, size=(n, 1))
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        quantile_values = np.quantile(y_true[:, 0], levels)
        y_quant = np.tile(quantile_values, (n, 1)).reshape(n, 1, len(levels))

        result = pit_coverage(y_true, y_quant, levels, target_names=["X"])
        for lv in levels:
            emp = result["aggregate"][lv]
            assert abs(emp - lv) < 0.01, f"coverage at {lv}={emp} not within 1pp"

    def test_miscalibrated_shifted_quantiles(self):
        # Shifted quantile predictions should diverge from nominal in a
        # predictable direction. Predicting q_0.5 too high (1.0 when true=0)
        # inflates empirical coverage; too low deflates it.
        rng = np.random.default_rng(1)
        n = 5000
        y_true = rng.normal(0.0, 1.0, size=(n, 1))

        q_too_high = np.full((n, 1, 1), 1.0)
        q_too_low = np.full((n, 1, 1), -1.0)

        r_high = pit_coverage(y_true, q_too_high, [0.5])
        r_low = pit_coverage(y_true, q_too_low, [0.5])
        # P(y<=1) ~ 0.84, P(y<=-1) ~ 0.16 — both far from nominal 0.5.
        assert r_high["aggregate"][0.5] > 0.7
        assert r_low["aggregate"][0.5] < 0.3

    def test_narrow_intervals_underperform_nominal(self):
        # True distribution N(0,1) has 50% central interval (-0.67, 0.67).
        # Predicting a narrow interval (-0.1, 0.1) captures only ~8% of mass,
        # far below the nominal 50% (nominal_0.75 - nominal_0.25).
        rng = np.random.default_rng(2)
        n = 10000
        y_true = rng.normal(0.0, 1.0, size=(n, 1))
        levels = [0.25, 0.75]
        q_narrow = np.stack(
            [np.full(n, -0.1), np.full(n, 0.1)],
            axis=-1,
        ).reshape(n, 1, 2)
        result = pit_coverage(y_true, q_narrow, levels)
        interval_cov = result["aggregate"][0.75] - result["aggregate"][0.25]
        assert interval_cov < 0.2, (
            f"Narrow intervals should capture far less than 0.5 mass; got {interval_cov}"
        )

    def test_output_per_target(self):
        y_true = np.array([[0.3, 0.4]] * 4)
        y_quant = np.zeros((4, 2, 2))
        # Levels 0.1 and 0.9: predict slightly below and above y for target 0
        y_quant[:, 0, 0] = 0.2  # below y_true[:, 0]=0.3
        y_quant[:, 0, 1] = 0.35  # above
        y_quant[:, 1, 0] = 0.5  # above y_true[:, 1]=0.4
        y_quant[:, 1, 1] = 0.5

        result = pit_coverage(y_true, y_quant, [0.1, 0.9], target_names=["OBP", "SLG"])
        # Target OBP: y=0.3, q_0.1=0.2 → coverage 0; q_0.9=0.35 → coverage 1
        assert result["per_target"]["OBP"][0.1] == pytest.approx(0.0)
        assert result["per_target"]["OBP"][0.9] == pytest.approx(1.0)
        # Target SLG: y=0.4, q_0.1=0.5 → coverage 1
        assert result["per_target"]["SLG"][0.1] == pytest.approx(1.0)

    def test_level_count_mismatch_raises(self):
        y_true = np.zeros((5, 2))
        y_quant = np.zeros((5, 2, 3))
        with pytest.raises(ValueError, match="3 quantiles but 2 levels"):
            pit_coverage(y_true, y_quant, [0.25, 0.75])


# ---------------------------------------------------------------------------
# pa_checkpoint_rows
# ---------------------------------------------------------------------------

class TestPaCheckpointRows:
    def _simple_snapshots(self) -> pd.DataFrame:
        """Two players across four iso-weeks with increasing pa_ytd."""
        rows = []
        # Player 1: reaches 50 at week 2, 100 at week 3, 200 at week 4
        for wk, pa in enumerate([30, 80, 150, 260], start=14):
            rows.append({"mlbam_id": 1, "season": 2024, "iso_year": 2024, "iso_week": wk, "pa_ytd": pa})
        # Player 2: reaches 50 at week 1, 100 at week 2, 200 at week 3, 400 at week 4
        for wk, pa in enumerate([60, 140, 230, 410], start=14):
            rows.append({"mlbam_id": 2, "season": 2024, "iso_year": 2024, "iso_week": wk, "pa_ytd": pa})
        return pd.DataFrame(rows)

    def test_first_crossing_per_threshold(self):
        df = self._simple_snapshots()
        result = pa_checkpoint_rows(df, thresholds=[50, 100, 200, 400])

        # 50 PA: player 1 at week 15, player 2 at week 14
        assert set(result[50]["mlbam_id"]) == {1, 2}
        p1_50 = result[50].loc[result[50]["mlbam_id"] == 1].iloc[0]
        p2_50 = result[50].loc[result[50]["mlbam_id"] == 2].iloc[0]
        assert p1_50["iso_week"] == 15 and p1_50["pa_ytd"] == 80
        assert p2_50["iso_week"] == 14 and p2_50["pa_ytd"] == 60

        # 400 PA: only player 2
        assert list(result[400]["mlbam_id"]) == [2]
        assert result[400].iloc[0]["pa_ytd"] == 410

    def test_missing_pa_col_raises(self):
        df = pd.DataFrame({"mlbam_id": [1], "season": [2024], "iso_year": [2024], "iso_week": [1]})
        with pytest.raises(KeyError, match="pa_ytd"):
            pa_checkpoint_rows(df)

    def test_missing_sort_col_raises(self):
        df = pd.DataFrame({"mlbam_id": [1], "pa_ytd": [60]})
        with pytest.raises(KeyError, match="missing sort columns"):
            pa_checkpoint_rows(df)

    def test_unsorted_input_handled(self):
        # Shuffle and confirm first-crossing uses chronological sort, not row order.
        df = self._simple_snapshots().sample(frac=1.0, random_state=0).reset_index(drop=True)
        result = pa_checkpoint_rows(df, thresholds=[100])
        p1 = result[100].loc[result[100]["mlbam_id"] == 1].iloc[0]
        # Player 1 first crosses 100 at week 16 (pa_ytd=150)
        assert p1["iso_week"] == 16 and p1["pa_ytd"] == 150

    def test_high_threshold_returns_empty(self):
        df = self._simple_snapshots()
        result = pa_checkpoint_rows(df, thresholds=[9999])
        assert len(result[9999]) == 0

    def test_one_row_per_player_season(self):
        df = self._simple_snapshots()
        result = pa_checkpoint_rows(df, thresholds=[50])
        # No duplicates per (mlbam_id, season)
        assert not result[50].duplicated(subset=["mlbam_id", "season"]).any()



"""Tests for the Bayesian shrinkage ROS baseline (Phase 1.3).

Covers:
- Beta-Binomial posterior-mean math (rates and count-per-PA stats)
- Per-stat ytd (successes, trials) extraction
- ``ShrinkageBaseline.predict`` end-to-end against the weekly-snapshot schema
- Tau-fitting via minimizing RMSE on held-out rows
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines.shrinkage import (
    DEFAULT_TAU0,
    ShrinkageBaseline,
    fit_tau_per_stat,
    shrinkage_posterior_mean,
    ytd_successes_trials,
)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


class TestShrinkagePosteriorMean:
    def test_closed_form_known_values(self):
        # tau0=100, prior=0.3 -> alpha0=30, beta0=70.
        # Observed 10 / 50 trials -> posterior mean = (30+10)/(100+50) = 40/150 = 0.2667.
        result = shrinkage_posterior_mean(
            preseason_rate=pd.Series([0.3]),
            successes=pd.Series([10.0]),
            trials=pd.Series([50.0]),
            tau0=100.0,
        )
        assert float(result.iloc[0]) == pytest.approx(40.0 / 150.0)

    def test_zero_trials_returns_prior(self):
        # No observations yet: posterior = prior.
        result = shrinkage_posterior_mean(
            preseason_rate=pd.Series([0.35]),
            successes=pd.Series([0.0]),
            trials=pd.Series([0.0]),
            tau0=100.0,
        )
        assert float(result.iloc[0]) == pytest.approx(0.35)

    def test_infinite_trials_dominates_prior(self):
        # Huge observed sample size: posterior -> observed rate.
        result = shrinkage_posterior_mean(
            preseason_rate=pd.Series([0.35]),
            successes=pd.Series([50000.0]),
            trials=pd.Series([100000.0]),
            tau0=100.0,
        )
        assert float(result.iloc[0]) == pytest.approx(0.5, abs=1e-3)

    def test_huge_tau_locks_to_prior(self):
        # tau0 >> observed trials: posterior ~ prior.
        result = shrinkage_posterior_mean(
            preseason_rate=pd.Series([0.30]),
            successes=pd.Series([50.0]),
            trials=pd.Series([100.0]),
            tau0=10_000.0,
        )
        assert float(result.iloc[0]) == pytest.approx(0.30, abs=1e-2)

    def test_vectorized(self):
        # Vector of players, all resolved element-wise.
        preseason = pd.Series([0.3, 0.35, 0.40])
        succ = pd.Series([10.0, 20.0, 30.0])
        trials = pd.Series([50.0, 80.0, 100.0])
        result = shrinkage_posterior_mean(preseason, succ, trials, tau0=100.0)
        expected = (100.0 * preseason + succ) / (100.0 + trials)
        pd.testing.assert_series_equal(result, expected, check_names=False)


# ---------------------------------------------------------------------------
# Per-stat successes / trials
# ---------------------------------------------------------------------------


def _row(**kwargs) -> pd.DataFrame:
    """Single-row DataFrame with sensible defaults for ytd counts."""
    defaults = dict(
        h_ytd=50.0, bb_ytd=20.0, hbp_ytd=3.0, sf_ytd=2.0,
        singles_ytd=30.0, doubles_ytd=12.0, triples_ytd=2.0,
        hr_ytd=6.0, r_ytd=30.0, rbi_ytd=28.0, sb_ytd=5.0,
        ab_ytd=180.0, pa_ytd=205.0,
    )
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


class TestYtdSuccessesTrials:
    def test_obp_formula(self):
        df = _row(h_ytd=50, bb_ytd=20, hbp_ytd=3, sf_ytd=2, ab_ytd=180)
        succ, trials = ytd_successes_trials(df, "obp")
        # OBP = (H+BB+HBP)/(AB+BB+HBP+SF) = 73 / 205 = 0.356
        assert float(succ.iloc[0]) == 73.0
        assert float(trials.iloc[0]) == 205.0

    def test_slg_formula(self):
        df = _row(singles_ytd=30, doubles_ytd=12, triples_ytd=2, hr_ytd=6, ab_ytd=180)
        succ, trials = ytd_successes_trials(df, "slg")
        # TB = 1B + 2*2B + 3*3B + 4*HR = 30 + 24 + 6 + 24 = 84
        assert float(succ.iloc[0]) == 84.0
        assert float(trials.iloc[0]) == 180.0

    def test_per_pa_stats_use_pa_as_trials(self):
        df = _row(hr_ytd=6, r_ytd=30, rbi_ytd=28, sb_ytd=5, pa_ytd=205)
        for stat, expected_succ in [("hr", 6), ("r", 30), ("rbi", 28), ("sb", 5)]:
            succ, trials = ytd_successes_trials(df, stat)
            assert float(succ.iloc[0]) == expected_succ, stat
            assert float(trials.iloc[0]) == 205, stat

    def test_nans_treated_as_zero(self):
        df = pd.DataFrame([{
            "h_ytd": np.nan, "bb_ytd": 10, "hbp_ytd": np.nan, "sf_ytd": 0,
            "ab_ytd": 100,
        }])
        succ, trials = ytd_successes_trials(df, "obp")
        assert float(succ.iloc[0]) == 10.0  # h=0, bb=10, hbp=0
        assert float(trials.iloc[0]) == 110.0

    def test_unknown_stat_raises(self):
        with pytest.raises(KeyError, match="Unsupported stat"):
            ytd_successes_trials(_row(), "unknown")


# ---------------------------------------------------------------------------
# ShrinkageBaseline predict
# ---------------------------------------------------------------------------


def _make_checkpoint_rows(n_players: int = 3, season: int = 2024) -> pd.DataFrame:
    rows = []
    for i in range(n_players):
        rows.append({
            "mlbam_id": 100 + i, "season": season,
            "iso_year": season, "iso_week": 20,
            "pa_ytd": 150.0 + 30.0 * i,
            "ab_ytd": 130.0 + 25.0 * i,
            "h_ytd": 40.0 + 5.0 * i, "bb_ytd": 15.0 + 2.0 * i,
            "hbp_ytd": 2.0, "sf_ytd": 1.0,
            "singles_ytd": 25.0 + 3.0 * i,
            "doubles_ytd": 10.0 + 1.0 * i,
            "triples_ytd": 1.0, "hr_ytd": 4.0 + 1.0 * i,
            "r_ytd": 20.0 + 4.0 * i,
            "rbi_ytd": 22.0 + 5.0 * i,
            "sb_ytd": 3.0 + i,
            "ros_obp": 0.34, "ros_slg": 0.45,
            "ros_hr_per_pa": 0.04, "ros_r_per_pa": 0.13,
            "ros_rbi_per_pa": 0.14, "ros_sb_per_pa": 0.025,
            "ros_pa": 300.0,
        })
    return pd.DataFrame(rows)


def _make_preseason_cache(n_players: int = 3, season: int = 2024) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "mlbam_id": 100 + i, "season": season,
            "target_obp": 0.300 + 0.02 * i, "target_slg": 0.400 + 0.03 * i,
            "target_hr": 0.035, "target_r": 0.13,
            "target_rbi": 0.14, "target_sb": 0.02,
        }
        for i in range(n_players)
    ])


class TestShrinkageBaselinePredict:
    def test_predict_returns_ros_rate_targets(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        model = ShrinkageBaseline()
        preds = model.predict(rows, preseason)

        from src.eval.ros_metrics import ROS_RATE_TARGETS
        assert preds is not None
        assert list(preds.columns) == list(ROS_RATE_TARGETS)
        assert len(preds) == len(rows)
        assert not preds.isna().any().any()

    def test_predict_matches_posterior_mean_per_stat(self):
        rows = _make_checkpoint_rows(n_players=1)
        preseason = _make_preseason_cache(n_players=1)
        tau_overrides = {"obp": 200.0, "slg": 200.0, "hr": 200.0, "r": 200.0, "rbi": 200.0, "sb": 200.0}
        model = ShrinkageBaseline(tau_per_stat=tau_overrides)
        preds = model.predict(rows, preseason)
        assert preds is not None

        # Recompute expected OBP shrinkage posterior mean by hand.
        succ_obp, trials_obp = ytd_successes_trials(rows, "obp")
        expected_obp = shrinkage_posterior_mean(
            pd.Series([0.300]), succ_obp, trials_obp, tau0=200.0,
        )
        assert float(preds["ros_obp"].iloc[0]) == pytest.approx(float(expected_obp.iloc[0]))

    def test_missing_preseason_cols_returns_none(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache().drop(columns=["target_sb"])
        model = ShrinkageBaseline()
        assert model.predict(rows, preseason) is None

    def test_missing_id_column_returns_none(self):
        rows = _make_checkpoint_rows().drop(columns=["mlbam_id"])
        preseason = _make_preseason_cache()
        model = ShrinkageBaseline()
        assert model.predict(rows, preseason) is None

    def test_zero_overlap_returns_none(self):
        # Stale preseason cache: no mlbam_id overlap.
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason["mlbam_id"] = preseason["mlbam_id"] + 10000
        model = ShrinkageBaseline()
        assert model.predict(rows, preseason) is None

    def test_unmatched_player_row_is_nan(self):
        rows = _make_checkpoint_rows(n_players=3)
        preseason = _make_preseason_cache(n_players=3)
        preseason = preseason[preseason["mlbam_id"] != 100]  # drop player 0
        model = ShrinkageBaseline()
        preds = model.predict(rows, preseason)
        # Player 0 (mlbam_id=100) has no prior, so its row must be all NaN;
        # players 1 and 2 must have valid posterior means.
        assert preds.iloc[0].isna().all()
        assert not preds.iloc[1].isna().any()
        assert not preds.iloc[2].isna().any()


# ---------------------------------------------------------------------------
# fit_tau_per_stat
# ---------------------------------------------------------------------------


class TestFitTauPerStat:
    def test_returns_dict_with_all_target_stats(self):
        rows = _make_checkpoint_rows(n_players=5)
        preseason = _make_preseason_cache(n_players=5)
        fitted = fit_tau_per_stat(rows, preseason)
        assert set(fitted) == set(DEFAULT_TAU0)
        for tau in fitted.values():
            assert tau > 0

    def test_minimum_near_known_optimum(self):
        # Build synthetic rows where the optimal tau for OBP is ~200:
        # each row's ros_obp equals the blend of a known preseason and ytd using tau=200.
        rng = np.random.default_rng(0)
        n = 200
        pa_ytd = rng.integers(100, 400, size=n).astype(float)
        ab_ytd = pa_ytd * 0.9
        obp_pre = rng.uniform(0.28, 0.40, size=n)
        obp_ytd = rng.uniform(0.24, 0.44, size=n)
        succ_obp = obp_ytd * pa_ytd
        tau_true = 200.0
        ros_obp_target = (tau_true * obp_pre + succ_obp) / (tau_true + pa_ytd)

        rows = pd.DataFrame({
            "mlbam_id": np.arange(n),
            "season": 2024,
            "pa_ytd": pa_ytd,
            "ab_ytd": ab_ytd,
            "h_ytd": succ_obp,   # for OBP: succ = H (bb/hbp/sf set to 0 below)
            "bb_ytd": 0.0,
            "hbp_ytd": 0.0,
            "sf_ytd": 0.0,
            "singles_ytd": 0.0, "doubles_ytd": 0.0, "triples_ytd": 0.0,
            "hr_ytd": 0.0, "r_ytd": 0.0, "rbi_ytd": 0.0, "sb_ytd": 0.0,
            "ros_obp": ros_obp_target,
            "ros_slg": 0.4, "ros_hr_per_pa": 0.03,
            "ros_r_per_pa": 0.11, "ros_rbi_per_pa": 0.12, "ros_sb_per_pa": 0.015,
        })
        # Adjust OBP trials: we set bb/hbp/sf = 0, so trials = ab_ytd; override to pa_ytd
        # by fudging ab to pa_ytd so succ/trials math uses our manufactured pa-based blend.
        rows["ab_ytd"] = pa_ytd

        preseason = pd.DataFrame({
            "mlbam_id": np.arange(n),
            "season": 2024,
            "target_obp": obp_pre,
            "target_slg": 0.4, "target_hr": 0.03,
            "target_r": 0.11, "target_rbi": 0.12, "target_sb": 0.015,
        })
        fitted = fit_tau_per_stat(rows, preseason)
        assert fitted["obp"] == pytest.approx(tau_true, rel=0.15)

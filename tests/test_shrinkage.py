"""Tests for the Bayesian shrinkage ROS baseline (Phase 1.3).

Covers:
- Beta-Binomial posterior-mean math (rates and count-per-PA stats)
- Per-stat ytd (successes, trials) extraction
- ``predict_shrinkage`` end-to-end against the weekly-snapshot schema
- Tau-fitting via minimizing RMSE on held-out rows
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.baselines.shrinkage import (
    DEFAULT_QUANTILE_TAUS,
    DEFAULT_TAU0,
    fit_tau_per_stat,
    predict_shrinkage,
    predict_shrinkage_quantiles,
    shrinkage_posterior_mean,
    shrinkage_posterior_quantiles,
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
        h_ytd=50.0,
        bb_ytd=20.0,
        hbp_ytd=3.0,
        sf_ytd=2.0,
        singles_ytd=30.0,
        doubles_ytd=12.0,
        triples_ytd=2.0,
        hr_ytd=6.0,
        r_ytd=30.0,
        rbi_ytd=28.0,
        sb_ytd=5.0,
        ab_ytd=180.0,
        pa_ytd=205.0,
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
        df = pd.DataFrame(
            [
                {
                    "h_ytd": np.nan,
                    "bb_ytd": 10,
                    "hbp_ytd": np.nan,
                    "sf_ytd": 0,
                    "ab_ytd": 100,
                }
            ]
        )
        succ, trials = ytd_successes_trials(df, "obp")
        assert float(succ.iloc[0]) == 10.0  # h=0, bb=10, hbp=0
        assert float(trials.iloc[0]) == 110.0

    def test_unknown_stat_raises(self):
        with pytest.raises(KeyError, match="Unsupported stat"):
            ytd_successes_trials(_row(), "unknown")


# ---------------------------------------------------------------------------
# predict_shrinkage
# ---------------------------------------------------------------------------


def _make_checkpoint_rows(n_players: int = 3, season: int = 2024) -> pd.DataFrame:
    rows = []
    for i in range(n_players):
        rows.append(
            {
                "mlbam_id": 100 + i,
                "season": season,
                "iso_year": season,
                "iso_week": 20,
                "pa_ytd": 150.0 + 30.0 * i,
                "ab_ytd": 130.0 + 25.0 * i,
                "h_ytd": 40.0 + 5.0 * i,
                "bb_ytd": 15.0 + 2.0 * i,
                "hbp_ytd": 2.0,
                "sf_ytd": 1.0,
                "singles_ytd": 25.0 + 3.0 * i,
                "doubles_ytd": 10.0 + 1.0 * i,
                "triples_ytd": 1.0,
                "hr_ytd": 4.0 + 1.0 * i,
                "r_ytd": 20.0 + 4.0 * i,
                "rbi_ytd": 22.0 + 5.0 * i,
                "sb_ytd": 3.0 + i,
                "ros_obp": 0.34,
                "ros_slg": 0.45,
                "ros_hr_per_pa": 0.04,
                "ros_r_per_pa": 0.13,
                "ros_rbi_per_pa": 0.14,
                "ros_sb_per_pa": 0.025,
                "ros_pa": 300.0,
            }
        )
    return pd.DataFrame(rows)


def _make_preseason_cache(n_players: int = 3, season: int = 2024) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mlbam_id": 100 + i,
                "season": season,
                "target_obp": 0.300 + 0.02 * i,
                "target_slg": 0.400 + 0.03 * i,
                "target_hr": 0.035,
                "target_r": 0.13,
                "target_rbi": 0.14,
                "target_sb": 0.02,
            }
            for i in range(n_players)
        ]
    )


class TestPredictShrinkage:
    def test_predict_returns_ros_rate_targets(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preds = predict_shrinkage(rows, preseason)

        from src.eval.ros_metrics import ROS_RATE_TARGETS

        assert preds is not None
        assert list(preds.columns) == list(ROS_RATE_TARGETS)
        assert len(preds) == len(rows)
        assert not preds.isna().any().any()

    def test_predict_matches_posterior_mean_per_stat(self):
        rows = _make_checkpoint_rows(n_players=1)
        preseason = _make_preseason_cache(n_players=1)
        tau_overrides = {
            "obp": 200.0,
            "slg": 200.0,
            "hr": 200.0,
            "r": 200.0,
            "rbi": 200.0,
            "sb": 200.0,
        }
        preds = predict_shrinkage(rows, preseason, tau_per_stat=tau_overrides)
        assert preds is not None

        # Recompute expected OBP shrinkage posterior mean by hand.
        succ_obp, trials_obp = ytd_successes_trials(rows, "obp")
        expected_obp = shrinkage_posterior_mean(
            pd.Series([0.300]),
            succ_obp,
            trials_obp,
            tau0=200.0,
        )
        assert float(preds["ros_obp"].iloc[0]) == pytest.approx(
            float(expected_obp.iloc[0])
        )

    def test_missing_preseason_cols_returns_none(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache().drop(columns=["target_sb"])
        assert predict_shrinkage(rows, preseason) is None

    def test_missing_id_column_returns_none(self):
        rows = _make_checkpoint_rows().drop(columns=["mlbam_id"])
        preseason = _make_preseason_cache()
        assert predict_shrinkage(rows, preseason) is None

    def test_zero_overlap_returns_none(self):
        # Stale preseason cache: no mlbam_id overlap.
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason["mlbam_id"] = preseason["mlbam_id"] + 10000
        assert predict_shrinkage(rows, preseason) is None

    def test_unmatched_player_row_is_nan(self):
        rows = _make_checkpoint_rows(n_players=3)
        preseason = _make_preseason_cache(n_players=3)
        preseason = preseason[preseason["mlbam_id"] != 100]  # drop player 0
        preds = predict_shrinkage(rows, preseason)
        # Player 0 (mlbam_id=100) has no prior, so its row must be all NaN;
        # players 1 and 2 must have valid posterior means.
        assert preds.iloc[0].isna().all()
        assert not preds.iloc[1].isna().any()
        assert not preds.iloc[2].isna().any()

    def test_multi_season_alignment_attaches_correct_year(self):
        # Same player in two seasons with deliberately different priors.
        # An id-only join would collapse both rows to the 2023 prior.
        rows_23 = _make_checkpoint_rows(n_players=1, season=2023)
        rows_24 = _make_checkpoint_rows(n_players=1, season=2024)
        rows = pd.concat([rows_23, rows_24], ignore_index=True)
        preseason = pd.DataFrame(
            [
                {
                    "mlbam_id": 100,
                    "season": 2023,
                    "target_obp": 0.30,
                    "target_slg": 0.40,
                    "target_hr": 0.02,
                    "target_r": 0.10,
                    "target_rbi": 0.11,
                    "target_sb": 0.01,
                },
                {
                    "mlbam_id": 100,
                    "season": 2024,
                    "target_obp": 0.37,
                    "target_slg": 0.50,
                    "target_hr": 0.07,
                    "target_r": 0.15,
                    "target_rbi": 0.16,
                    "target_sb": 0.04,
                },
            ]
        )
        # Use a huge tau0 so posterior mean ≈ prior — makes the alignment
        # easy to verify by looking at the output.
        tau_overrides = dict.fromkeys(DEFAULT_TAU0, 1e6)
        preds = predict_shrinkage(rows, preseason, tau_per_stat=tau_overrides)
        assert preds is not None
        assert float(preds["ros_obp"].iloc[0]) == pytest.approx(0.30, abs=1e-3)
        assert float(preds["ros_obp"].iloc[1]) == pytest.approx(0.37, abs=1e-3)


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

    def test_multi_year_joins_on_id_and_season(self):
        # Same player in two seasons with very different priors. The 2023
        # prior is close to the 2023 ROS outcome; the 2024 prior is way off.
        # If the fit erroneously drops-by-id only, one year's observations get
        # matched to the wrong year's prior, and the fit degrades noticeably.
        n_per_year = 40
        rng = np.random.default_rng(3)

        def year_rows(
            year: int, pre_obp: float, ros_obp_target: float
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
            pa = rng.integers(150, 350, size=n_per_year).astype(float)
            ab = (
                pa.copy()
            )  # succ = H, trials = AB+BB+HBP+SF; set others to 0 so trials == ab
            succ = rng.uniform(0.28, 0.38, size=n_per_year) * pa
            rows = pd.DataFrame(
                {
                    "mlbam_id": np.arange(n_per_year),
                    "season": year,
                    "pa_ytd": pa,
                    "ab_ytd": ab,
                    "h_ytd": succ,
                    "bb_ytd": 0.0,
                    "hbp_ytd": 0.0,
                    "sf_ytd": 0.0,
                    "singles_ytd": 0.0,
                    "doubles_ytd": 0.0,
                    "triples_ytd": 0.0,
                    "hr_ytd": 0.0,
                    "r_ytd": 0.0,
                    "rbi_ytd": 0.0,
                    "sb_ytd": 0.0,
                    "ros_obp": ros_obp_target + rng.normal(0, 0.005, n_per_year),
                    "ros_slg": 0.4,
                    "ros_hr_per_pa": 0.03,
                    "ros_r_per_pa": 0.11,
                    "ros_rbi_per_pa": 0.12,
                    "ros_sb_per_pa": 0.015,
                }
            )
            pre = pd.DataFrame(
                {
                    "mlbam_id": np.arange(n_per_year),
                    "season": year,
                    "target_obp": pre_obp,
                    "target_slg": 0.4,
                    "target_hr": 0.03,
                    "target_r": 0.11,
                    "target_rbi": 0.12,
                    "target_sb": 0.015,
                }
            )
            return rows, pre

        rows_23, pre_23 = year_rows(2023, pre_obp=0.32, ros_obp_target=0.32)
        rows_24, pre_24 = year_rows(2024, pre_obp=0.36, ros_obp_target=0.36)
        training_rows = pd.concat([rows_23, rows_24], ignore_index=True)
        preseason = pd.concat([pre_23, pre_24], ignore_index=True)

        fitted = fit_tau_per_stat(training_rows, preseason, stats=["obp"])
        # With a correct season-aware join, 80 rows survive and each gets the
        # right-year prior (which happens to match their ros_obp target), so
        # the optimal tau is essentially infinite → fitted should hit the
        # upper search bound. With the buggy id-only join, only 40 rows
        # survive (one year's prior gets applied to both years' observations),
        # introducing systematic error that pulls tau toward the middle.
        assert fitted["obp"] > 2000.0, (
            f"Expected per-year priors to each match their ros target, "
            f"producing tau -> upper bound; got {fitted['obp']}. "
            f"Likely joining on mlbam_id only and losing the season dimension."
        )

    def test_falls_back_when_preseason_missing_stat(self):
        rows = _make_checkpoint_rows(n_players=5)
        preseason = _make_preseason_cache(n_players=5).drop(columns=["target_sb"])
        fitted = fit_tau_per_stat(rows, preseason)
        assert fitted["sb"] == pytest.approx(DEFAULT_TAU0["sb"])
        # Non-missing stats still get fit (produce positive finite values).
        for other in ("obp", "slg", "hr", "r", "rbi"):
            assert fitted[other] > 0

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

        rows = pd.DataFrame(
            {
                "mlbam_id": np.arange(n),
                "season": 2024,
                "pa_ytd": pa_ytd,
                "ab_ytd": ab_ytd,
                "h_ytd": succ_obp,  # for OBP: succ = H (bb/hbp/sf set to 0 below)
                "bb_ytd": 0.0,
                "hbp_ytd": 0.0,
                "sf_ytd": 0.0,
                "singles_ytd": 0.0,
                "doubles_ytd": 0.0,
                "triples_ytd": 0.0,
                "hr_ytd": 0.0,
                "r_ytd": 0.0,
                "rbi_ytd": 0.0,
                "sb_ytd": 0.0,
                "ros_obp": ros_obp_target,
                "ros_slg": 0.4,
                "ros_hr_per_pa": 0.03,
                "ros_r_per_pa": 0.11,
                "ros_rbi_per_pa": 0.12,
                "ros_sb_per_pa": 0.015,
            }
        )
        # Adjust OBP trials: we set bb/hbp/sf = 0, so trials = ab_ytd; override to pa_ytd
        # by fudging ab to pa_ytd so succ/trials math uses our manufactured pa-based blend.
        rows["ab_ytd"] = pa_ytd

        preseason = pd.DataFrame(
            {
                "mlbam_id": np.arange(n),
                "season": 2024,
                "target_obp": obp_pre,
                "target_slg": 0.4,
                "target_hr": 0.03,
                "target_r": 0.11,
                "target_rbi": 0.12,
                "target_sb": 0.015,
            }
        )
        fitted = fit_tau_per_stat(rows, preseason)
        assert fitted["obp"] == pytest.approx(tau_true, rel=0.15)


# ---------------------------------------------------------------------------
# Beta posterior quantiles
# ---------------------------------------------------------------------------


class TestShrinkagePosteriorQuantiles:
    def test_beta_quantile_monotonicity(self):
        # Quantiles must be non-decreasing across tau, element-wise.
        taus = (0.05, 0.25, 0.50, 0.75, 0.95)
        q = shrinkage_posterior_quantiles(
            preseason_rate=pd.Series([0.30, 0.35, 0.40]),
            successes=pd.Series([10.0, 20.0, 30.0]),
            trials=pd.Series([50.0, 80.0, 100.0]),
            tau0=100.0,
            taus=taus,
        )
        arr = q.to_numpy()
        # Across the tau axis, each row should be non-decreasing.
        diffs = np.diff(arr, axis=1)
        assert np.all(diffs >= -1e-9), (
            f"Beta quantiles must be non-decreasing across tau; got diffs {diffs}"
        )

    def test_beta_quantile_median_matches_ppf_05(self):
        # With successes chosen so alpha == beta, the Beta is symmetric and
        # the median coincides with the mean (both at 0.5). The posterior
        # mean formula is (tau0*p + s) / (tau0 + t). Pick tau0=100, p=0.5,
        # s=25, t=50 → alpha = 50+25 = 75, beta = 50 + 25 = 75 → symmetric.
        q = shrinkage_posterior_quantiles(
            preseason_rate=pd.Series([0.5]),
            successes=pd.Series([25.0]),
            trials=pd.Series([50.0]),
            tau0=100.0,
            taus=(0.5,),
        )
        assert float(q.iloc[0, 0]) == pytest.approx(0.5, abs=1e-6)

    def test_beta_quantile_bounds_in_unit_interval(self):
        # All tau quantiles are in [0, 1] — Beta is supported on [0, 1].
        rng = np.random.default_rng(0)
        n = 50
        q = shrinkage_posterior_quantiles(
            preseason_rate=pd.Series(rng.uniform(0.02, 0.6, n)),
            successes=pd.Series(rng.uniform(0.0, 200.0, n)),
            trials=pd.Series(rng.uniform(50.0, 500.0, n)),
            tau0=150.0,
            taus=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
        )
        arr = q.to_numpy()
        assert (arr >= 0.0).all()
        assert (arr <= 1.0).all()

    def test_beta_quantile_shape(self):
        # Output has exactly one column per tau and preserves row count.
        q = shrinkage_posterior_quantiles(
            preseason_rate=pd.Series([0.3, 0.35, 0.4]),
            successes=pd.Series([10.0, 20.0, 30.0]),
            trials=pd.Series([50.0, 80.0, 100.0]),
            tau0=100.0,
            taus=DEFAULT_QUANTILE_TAUS,
        )
        assert q.shape == (3, len(DEFAULT_QUANTILE_TAUS))
        # Column names follow the q{tau}_{} convention.
        expected_cols = [f"q{t:.2f}".replace(".", "_") for t in DEFAULT_QUANTILE_TAUS]
        assert list(q.columns) == expected_cols

    def test_beta_quantile_clips_nonpositive_params(self):
        # A negative preseason rate (edge case from noisy MTL output) with
        # no observations could otherwise produce alpha<=0 and ppf=NaN.
        # The implementation clips to a small positive floor so scipy returns
        # finite quantile values rather than NaN.
        q = shrinkage_posterior_quantiles(
            preseason_rate=pd.Series([-0.2, 1.3]),
            successes=pd.Series([0.0, 0.0]),
            trials=pd.Series([0.0, 0.0]),
            tau0=100.0,
            taus=(0.25, 0.5, 0.75),
        )
        assert q.notna().all(axis=None)


class TestPredictShrinkageQuantiles:
    def test_returns_dict_keyed_by_ros_target(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preds = predict_shrinkage_quantiles(rows, preseason)

        from src.eval.ros_metrics import ROS_RATE_TARGETS

        assert preds is not None
        assert set(preds) == set(ROS_RATE_TARGETS)
        # Each value is a DataFrame with one column per tau and one row per checkpoint.
        for target, df in preds.items():
            assert df.shape == (len(rows), len(DEFAULT_QUANTILE_TAUS))

    def test_predict_shrinkage_quantiles_none_on_missing_preseason(self):
        # Missing preseason target column → alignment fails → return None.
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache().drop(columns=["target_sb"])
        assert predict_shrinkage_quantiles(rows, preseason) is None

    def test_predict_shrinkage_quantiles_none_on_missing_id(self):
        rows = _make_checkpoint_rows().drop(columns=["mlbam_id"])
        preseason = _make_preseason_cache()
        assert predict_shrinkage_quantiles(rows, preseason) is None

    def test_predict_shrinkage_quantiles_none_on_zero_overlap(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason["mlbam_id"] = preseason["mlbam_id"] + 10000
        assert predict_shrinkage_quantiles(rows, preseason) is None

    def test_quantiles_consistent_with_posterior_mean(self):
        # Median (tau=0.5) of the Beta posterior should be close to the
        # posterior mean when alpha and beta are both reasonably large
        # (Beta mean vs median differ by O(1/n) for symmetric-ish cases).
        rows = _make_checkpoint_rows(n_players=1)
        preseason = _make_preseason_cache(n_players=1)
        mean_preds = predict_shrinkage(rows, preseason)
        quantile_preds = predict_shrinkage_quantiles(rows, preseason, taus=(0.5,))
        assert mean_preds is not None
        assert quantile_preds is not None
        for target in mean_preds.columns:
            m = float(mean_preds[target].iloc[0])
            q50 = float(quantile_preds[target].iloc[0, 0])
            # Allow some skew — Beta median != mean in general; 0.05 is
            # plenty for the integer-ish parameters in our fixtures.
            assert q50 == pytest.approx(m, abs=0.05)

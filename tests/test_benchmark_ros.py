"""Tests for benchmark_ros baseline predictors + evaluation flow.

Focuses on pure-Python behavior (no model training, no file I/O beyond
small parquet round-trips).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# `scripts/benchmark_ros.py` lives outside the src/ package; import directly.
import benchmark_ros as br  # noqa: E402
from src.eval.ros_metrics import ROS_RATE_TARGETS, ROS_YTD_RATES  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_checkpoint_rows(n_players: int = 3, season: int = 2024) -> pd.DataFrame:
    """Build a minimal checkpoint DataFrame with ytd + ROS target columns."""
    rows = []
    for i in range(n_players):
        pa = 150.0 + 30.0 * i
        ab = pa * 0.9
        rows.append({
            "mlbam_id": 100 + i,
            "season": season,
            "iso_year": season,
            "iso_week": 20,
            "pa_ytd": pa,
            # YTD rates (inputs)
            "obp_ytd": 0.32 + 0.01 * i,
            "slg_ytd": 0.42 + 0.01 * i,
            "hr_per_pa_ytd": 0.03 + 0.005 * i,
            "r_per_pa_ytd": 0.12 + 0.005 * i,
            "rbi_per_pa_ytd": 0.13 + 0.005 * i,
            "sb_per_pa_ytd": 0.02 + 0.005 * i,
            # YTD counts (needed by the shrinkage baseline)
            "ab_ytd": ab,
            "h_ytd": ab * (0.27 + 0.005 * i),
            "bb_ytd": pa * (0.08 + 0.005 * i),
            "hbp_ytd": pa * 0.01,
            "sf_ytd": pa * 0.005,
            "singles_ytd": ab * 0.18,
            "doubles_ytd": ab * 0.05,
            "triples_ytd": ab * 0.005,
            "hr_ytd": pa * (0.03 + 0.005 * i),
            "r_ytd": pa * (0.12 + 0.005 * i),
            "rbi_ytd": pa * (0.13 + 0.005 * i),
            "sb_ytd": pa * (0.02 + 0.005 * i),
            # ROS targets (outcomes)
            "ros_obp": 0.34 + 0.01 * i,
            "ros_slg": 0.45 + 0.01 * i,
            "ros_hr_per_pa": 0.04 + 0.005 * i,
            "ros_r_per_pa": 0.13 + 0.005 * i,
            "ros_rbi_per_pa": 0.14 + 0.005 * i,
            "ros_sb_per_pa": 0.025 + 0.005 * i,
            "ros_pa": 300,
        })
    return pd.DataFrame(rows)


def _make_preseason_cache(n_players: int = 3, season: int = 2024) -> pd.DataFrame:
    """Preseason MTL predictions matching the checkpoint players."""
    rows = []
    for i in range(n_players):
        rows.append({
            "mlbam_id": 100 + i,
            "idfg": 200 + i,
            "season": season,
            # Rate targets — preseason predictions differ from ytd rates.
            "target_obp": 0.30 + 0.015 * i,
            "target_slg": 0.40 + 0.015 * i,
            "target_hr": 0.035,
            "target_r": 0.11,
            "target_rbi": 0.12,
            "target_sb": 0.015,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline predictors
# ---------------------------------------------------------------------------


class TestPersistObserved:
    def test_predictions_equal_ytd_rates(self):
        rows = _make_checkpoint_rows()
        preds = br.predict_persist_observed(rows)

        assert list(preds.columns) == list(ROS_RATE_TARGETS)
        # Column i in preds should equal the corresponding ytd rate column
        for tgt, ytd in zip(ROS_RATE_TARGETS, ROS_YTD_RATES):
            np.testing.assert_allclose(preds[tgt].values, rows[ytd].values)


class TestFrozenPreseason:
    def test_broadcasts_preseason_by_mlbam_id(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preds = br.predict_frozen_preseason(rows, preseason)

        assert preds is not None
        assert list(preds.columns) == list(ROS_RATE_TARGETS)
        # ros_obp prediction equals preseason target_obp, row-aligned
        for i in range(len(rows)):
            expected = preseason.loc[preseason["mlbam_id"] == rows.iloc[i]["mlbam_id"], "target_obp"].iloc[0]
            assert preds.iloc[i]["ros_obp"] == pytest.approx(expected)

    def test_missing_id_column_returns_none(self):
        rows = _make_checkpoint_rows().drop(columns=["mlbam_id"])
        preseason = _make_preseason_cache()
        assert br.predict_frozen_preseason(rows, preseason) is None

    def test_missing_preseason_columns_returns_none(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache().drop(columns=["target_sb"])
        assert br.predict_frozen_preseason(rows, preseason) is None

    def test_unmatched_player_nan(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason = preseason[preseason["mlbam_id"] != 100]
        preds = br.predict_frozen_preseason(rows, preseason)
        # Row for mlbam_id=100 should be entirely NaN
        assert preds.iloc[0].isna().all()
        # Row for mlbam_id=101 should be populated
        assert not preds.iloc[1].isna().any()

    def test_zero_overlap_returns_none(self):
        # Preseason cache for different players than the checkpoint rows —
        # e.g. stale cache or dtype mismatch — should yield None, not an
        # all-NaN frame that would wipe every baseline in evaluate_checkpoint.
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason["mlbam_id"] = preseason["mlbam_id"] + 10000
        assert br.predict_frozen_preseason(rows, preseason) is None


class TestMarcelBlend:
    def test_blend_formula(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        prior = 200.0
        preds = br.predict_marcel_blend(rows, preseason, prior_pa=prior)

        assert preds is not None
        # Row 0: pa_ytd=150, prior=200 -> w_obs=150/350=0.4286
        ytd_pa = rows.iloc[0]["pa_ytd"]
        w_obs = ytd_pa / (ytd_pa + prior)
        w_pre = prior / (ytd_pa + prior)
        expected_obp = rows.iloc[0]["obp_ytd"] * w_obs + preseason.iloc[0]["target_obp"] * w_pre
        assert preds.iloc[0]["ros_obp"] == pytest.approx(expected_obp)

    def test_zero_pa_yields_pure_prior(self):
        rows = _make_checkpoint_rows(n_players=1)
        rows["pa_ytd"] = 0
        preseason = _make_preseason_cache(n_players=1)
        preds = br.predict_marcel_blend(rows, preseason, prior_pa=200.0)
        assert preds.iloc[0]["ros_obp"] == pytest.approx(preseason.iloc[0]["target_obp"])

    def test_huge_pa_dominates_prior(self):
        rows = _make_checkpoint_rows(n_players=1)
        rows["pa_ytd"] = 100000
        preseason = _make_preseason_cache(n_players=1)
        preds = br.predict_marcel_blend(rows, preseason, prior_pa=200.0)
        assert preds.iloc[0]["ros_obp"] == pytest.approx(rows.iloc[0]["obp_ytd"], abs=1e-4)

    def test_missing_preseason_returns_none(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache().drop(columns=["target_obp"])
        assert br.predict_marcel_blend(rows, preseason) is None


class TestShrinkagePredictor:
    def test_baseline_registered(self):
        # The shrinkage baseline must be part of the preseason-dependent set
        # so evaluate_checkpoint dispatches it correctly.
        assert "shrinkage" in br.ALL_BASELINES
        assert "shrinkage" in br._BASELINES_NEED_PRESEASON

    def test_predict_returns_dataframe(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preds = br.predict_shrinkage(rows, preseason)
        assert preds is not None
        assert list(preds.columns) == list(ROS_RATE_TARGETS)
        assert not preds.isna().any().any()

    def test_zero_overlap_returns_none(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason["mlbam_id"] = preseason["mlbam_id"] + 10000
        assert br.predict_shrinkage(rows, preseason) is None


# ---------------------------------------------------------------------------
# evaluate_checkpoint
# ---------------------------------------------------------------------------


class TestEvaluateCheckpoint:
    def test_persist_observed_only(self):
        rows = _make_checkpoint_rows()
        result = br.evaluate_checkpoint(
            rows,
            baselines=["persist_observed"],
            preseason=None,
            prior_pa=200.0,
            min_ros_pa=50,
        )
        assert result["n_players"] == 3
        assert set(result["systems"]) == {"persist_observed"}

    def test_all_baselines_with_preseason(self):
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        result = br.evaluate_checkpoint(
            rows,
            baselines=list(br.ALL_BASELINES),
            preseason=preseason,
            prior_pa=200.0,
            min_ros_pa=50,
        )
        assert result["n_players"] == 3
        assert set(result["systems"]) == {
            "persist_observed", "frozen_preseason", "marcel_blend", "shrinkage",
        }

    def test_filters_low_ros_pa(self):
        rows = _make_checkpoint_rows(n_players=4)
        rows.loc[:1, "ros_pa"] = 10  # two rows below min
        result = br.evaluate_checkpoint(
            rows, baselines=["persist_observed"],
            preseason=None, prior_pa=200.0, min_ros_pa=50,
        )
        assert result["n_players"] == 2

    def test_drops_rows_with_nan_target(self):
        rows = _make_checkpoint_rows()
        rows.loc[0, "ros_obp"] = np.nan
        result = br.evaluate_checkpoint(
            rows, baselines=["persist_observed"],
            preseason=None, prior_pa=200.0, min_ros_pa=50,
        )
        assert result["n_players"] == 2

    def test_empty_filter_returns_zero(self):
        rows = _make_checkpoint_rows()
        rows["ros_pa"] = 0
        result = br.evaluate_checkpoint(
            rows, baselines=["persist_observed"],
            preseason=None, prior_pa=200.0, min_ros_pa=50,
        )
        assert result == {"n_players": 0, "systems": {}}

    def test_baseline_missing_preseason_skipped(self):
        rows = _make_checkpoint_rows()
        # Request frozen_preseason without providing preseason data
        result = br.evaluate_checkpoint(
            rows,
            baselines=["persist_observed", "frozen_preseason"],
            preseason=None,
            prior_pa=200.0,
            min_ros_pa=50,
        )
        # Only persist_observed should run
        assert set(result["systems"]) == {"persist_observed"}

    def test_zero_overlap_preseason_preserves_persist_observed(self):
        # Stale preseason cache whose IDs don't match any checkpoint row must
        # NOT wipe persist_observed results (the original bug Codex caught).
        rows = _make_checkpoint_rows()
        preseason = _make_preseason_cache()
        preseason["mlbam_id"] = preseason["mlbam_id"] + 10000
        result = br.evaluate_checkpoint(
            rows,
            baselines=list(br.ALL_BASELINES),
            preseason=preseason,
            prior_pa=200.0,
            min_ros_pa=50,
        )
        assert result["n_players"] == 3
        assert set(result["systems"]) == {"persist_observed"}


# ---------------------------------------------------------------------------
# pool_by_threshold
# ---------------------------------------------------------------------------


class TestPoolByThreshold:
    def _mk_year_result(self, year: int, threshold: int, n_samples: int = 5) -> dict:
        rng = np.random.default_rng(year)
        y_true = rng.uniform(0.2, 0.5, size=(n_samples, 6))
        y_pred_a = y_true + rng.normal(0, 0.01, size=y_true.shape)
        y_pred_b = y_true + rng.normal(0, 0.02, size=y_true.shape)
        # Metrics would normally be computed by evaluate_checkpoint; stub them here
        return {
            "year": year,
            "thresholds": {
                threshold: {
                    "n_players": n_samples,
                    "y_true": y_true,
                    "systems": {
                        "persist_observed": {"y_pred": y_pred_a, "metrics": {}},
                        "marcel_blend": {"y_pred": y_pred_b, "metrics": {}},
                    },
                },
            },
        }

    def test_pools_across_years(self):
        yrs = [self._mk_year_result(2023, 100), self._mk_year_result(2024, 100)]
        pooled = br.pool_by_threshold(yrs)
        assert set(pooled) == {100}
        assert set(pooled[100]["systems"]) == {"persist_observed", "marcel_blend"}
        # n_players_per_system reflects concatenated size
        assert pooled[100]["n_players_per_system"]["persist_observed"] == 10

    def test_empty_results(self):
        assert br.pool_by_threshold([]) == {}


# ---------------------------------------------------------------------------
# load_weekly_snapshots
# ---------------------------------------------------------------------------


class TestLoadWeeklySnapshots:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build_snapshots"):
            br.load_weekly_snapshots([2099], raw_dir=tmp_path)

    def test_concatenates_years(self, tmp_path):
        df_a = pd.DataFrame({"mlbam_id": [1], "season": [2023], "pa_ytd": [100]})
        df_b = pd.DataFrame({"mlbam_id": [2], "season": [2024], "pa_ytd": [200]})
        df_a.to_parquet(tmp_path / "weekly_snapshots_2023.parquet")
        df_b.to_parquet(tmp_path / "weekly_snapshots_2024.parquet")
        combined = br.load_weekly_snapshots([2024, 2023], raw_dir=tmp_path)
        assert len(combined) == 2
        assert set(combined["season"]) == {2023, 2024}


# ---------------------------------------------------------------------------
# Preseason cache round-trip
# ---------------------------------------------------------------------------


class TestPreseasonCache:
    def test_loads_existing_cache(self, tmp_path):
        preseason = _make_preseason_cache()
        cache_path = tmp_path / "mtl_preseason_2024.parquet"
        preseason.to_parquet(cache_path)

        loaded = br.load_or_generate_preseason_cache(
            year=2024,
            cache_dir=tmp_path,
            df_featured=None,
            data_config=None,
            retrain=False,
        )
        assert loaded is not None
        assert len(loaded) == len(preseason)

    def test_missing_cache_no_retrain_returns_none(self, tmp_path):
        result = br.load_or_generate_preseason_cache(
            year=2024,
            cache_dir=tmp_path,
            df_featured=None,
            data_config=None,
            retrain=False,
        )
        assert result is None

    def test_retrain_without_inputs_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="df_featured"):
            br.load_or_generate_preseason_cache(
                year=2024,
                cache_dir=tmp_path,
                df_featured=None,
                data_config=None,
                retrain=True,
            )

    def test_retrain_bypasses_existing_cache(self, tmp_path):
        # Existing cache + retrain=True must NOT short-circuit; otherwise the
        # cache stays stale after feature/model changes. Retraining without
        # df_featured should reach the same error path as a missing cache.
        _make_preseason_cache().to_parquet(tmp_path / "mtl_preseason_2024.parquet")
        with pytest.raises(RuntimeError, match="df_featured"):
            br.load_or_generate_preseason_cache(
                year=2024,
                cache_dir=tmp_path,
                df_featured=None,
                data_config=None,
                retrain=True,
            )

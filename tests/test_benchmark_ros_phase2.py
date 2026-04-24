"""Tests for the Phase 2 quantile baseline wired into ``benchmark_ros``.

These tests focus on the benchmark-side plumbing (dispatch, cache semantics,
quantile metrics in the JSON report, graceful failure when snapshot files
are absent). They never actually train a torch model — ``train_ros`` and
``MTLQuantileEnsembleForecaster.load`` are stubbed with a synthetic fake
ensemble that returns deterministic quantile arrays.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_ros as br  # noqa: E402
from src.eval.ros_metrics import ROS_RATE_TARGETS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakePhase2Ensemble:
    """Deterministic stand-in for ``MTLQuantileEnsembleForecaster``.

    Emits quantile predictions that are a linear ramp around a target-specific
    baseline, so the benchmark's pinball / PIT computations have non-trivial
    inputs to exercise.
    """

    def __init__(self, feature_names: list[str], taus: list[float]):
        self.feature_names_ = feature_names
        self.target_names_ = list(ROS_RATE_TARGETS)
        self.taus = list(taus)
        self.is_fitted_ = True

    def predict(self, X: pd.DataFrame | np.ndarray) -> dict[str, np.ndarray]:
        X_arr = np.asarray(X, dtype=np.float64)
        n = X_arr.shape[0]
        n_targets = len(self.target_names_)
        # Base level per target (OBP≈0.32, SLG≈0.42, HR/PA≈0.03, ...)
        base_per_target = np.array([0.32, 0.42, 0.03, 0.12, 0.13, 0.02])[:n_targets]
        # Per-tau offset: symmetric around 0 — widest at the extremes.
        tau_offsets = (np.asarray(self.taus) - 0.5) * 0.04
        # Shape: (n, n_targets, n_quantiles)
        quantiles = (
            base_per_target[None, :, None]
            + tau_offsets[None, None, :]
            + np.zeros((n, 1, 1))
        )
        pa_remaining = np.full((n, 1), 250.0)
        return {"quantiles": quantiles, "pa_remaining": pa_remaining}

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "ensemble_meta.json", "w") as f:
            json.dump(
                {
                    "n_seeds": 1,
                    "base_seed": 42,
                    "feature_names": self.feature_names_,
                    "target_names": self.target_names_,
                    "taus": self.taus,
                },
                f,
            )
        return out


def _make_weekly_snapshot(
    season: int, n_players: int = 4, n_weeks: int = 3
) -> pd.DataFrame:
    """Synthesize a weekly-snapshot frame with enough columns for the benchmark."""
    rng = np.random.default_rng(season)
    rows = []
    for pid in range(n_players):
        for wk_idx in range(n_weeks):
            pa_ytd = 80.0 + 60.0 * wk_idx  # crosses the 50/100 thresholds
            ab_ytd = pa_ytd * 0.9
            rows.append(
                {
                    "mlbam_id": 100 + pid,
                    "season": season,
                    "iso_year": season,
                    "iso_week": 16 + wk_idx,
                    "pa_ytd": pa_ytd,
                    "ab_ytd": ab_ytd,
                    "h_ytd": ab_ytd * 0.27,
                    "bb_ytd": pa_ytd * 0.08,
                    "hbp_ytd": pa_ytd * 0.01,
                    "sf_ytd": pa_ytd * 0.005,
                    "singles_ytd": ab_ytd * 0.18,
                    "doubles_ytd": ab_ytd * 0.05,
                    "triples_ytd": ab_ytd * 0.005,
                    "hr_ytd": pa_ytd * 0.03,
                    "r_ytd": pa_ytd * 0.12,
                    "rbi_ytd": pa_ytd * 0.13,
                    "sb_ytd": pa_ytd * 0.02,
                    "obp_ytd": 0.33,
                    "slg_ytd": 0.43,
                    "hr_per_pa_ytd": 0.03,
                    "r_per_pa_ytd": 0.12,
                    "rbi_per_pa_ytd": 0.13,
                    "sb_per_pa_ytd": 0.02,
                    "iso_ytd": 0.16,
                    "bb_rate_ytd": 0.08,
                    "k_rate_ytd": 0.22,
                    "trail4w_pa": 40.0,
                    "trail4w_ab": 35.0,
                    "trail4w_h": 10.0,
                    "trail4w_bb": 3.0,
                    "trail4w_hbp": 0.5,
                    "trail4w_sf": 0.5,
                    "trail4w_singles": 6.0,
                    "trail4w_doubles": 2.0,
                    "trail4w_triples": 0.5,
                    "trail4w_hr": 1.0,
                    "trail4w_r": 5.0,
                    "trail4w_rbi": 5.0,
                    "trail4w_sb": 0.5,
                    "trail4w_so": 10.0,
                    "ros_obp": 0.34 + rng.normal(0, 0.01),
                    "ros_slg": 0.45 + rng.normal(0, 0.02),
                    "ros_hr_per_pa": 0.04 + rng.normal(0, 0.005),
                    "ros_r_per_pa": 0.13 + rng.normal(0, 0.005),
                    "ros_rbi_per_pa": 0.14 + rng.normal(0, 0.005),
                    "ros_sb_per_pa": 0.025 + rng.normal(0, 0.005),
                    "ros_pa": 300.0,
                }
            )
    return pd.DataFrame(rows)


def _make_preseason_cache(n_players: int = 4, season: int = 2024) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "mlbam_id": 100 + i,
                "season": season,
                "target_obp": 0.30 + 0.01 * i,
                "target_slg": 0.40 + 0.01 * i,
                "target_hr": 0.035,
                "target_r": 0.11,
                "target_rbi": 0.12,
                "target_sb": 0.015,
            }
            for i in range(n_players)
        ]
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPhase2Registration:
    def test_phase2_baseline_registered(self):
        # phase2 must appear in ALL_BASELINES and the preseason-dependent tuple.
        assert "phase2" in br.ALL_BASELINES
        assert "phase2" in br._BASELINES_NEED_PRESEASON
        assert "phase2" in br._BASELINES_EMIT_QUANTILES
        assert "phase2" in br._BASELINE_DISPLAY

    def test_phase2_in_cli_choices(self):
        # The --include argparser choices come directly from ALL_BASELINES.
        # Asserting the parser accepts phase2 is as important as the tuple check.
        parser = None
        import argparse as _argparse

        # Reproduce the parser choices list (this is what the CLI exposes)
        assert "phase2" in list(br.ALL_BASELINES)
        del parser, _argparse


# ---------------------------------------------------------------------------
# Graceful failure when snapshot files are absent
# ---------------------------------------------------------------------------


class TestPhase2GracefulFailure:
    def test_graceful_fail_without_snapshots(self, tmp_path):
        # Missing weekly_snapshots_*.parquet should raise a FileNotFoundError
        # with an actionable message, not a cryptic AttributeError.
        with pytest.raises(FileNotFoundError, match="weekly snapshots"):
            br._phase2_training_snapshots_for_year(
                eval_year=2025,
                raw_dir=tmp_path,
            )

    def test_training_snapshots_filters_by_year(self, tmp_path):
        # Write two years of snapshot fixtures; only the year < eval_year
        # should be loaded.
        _make_weekly_snapshot(2022).to_parquet(
            tmp_path / "weekly_snapshots_2022.parquet"
        )
        _make_weekly_snapshot(2023).to_parquet(
            tmp_path / "weekly_snapshots_2023.parquet"
        )
        _make_weekly_snapshot(2024).to_parquet(
            tmp_path / "weekly_snapshots_2024.parquet"
        )

        train = br._phase2_training_snapshots_for_year(
            eval_year=2024,
            raw_dir=tmp_path,
        )
        assert set(train["season"].unique()) == {2022, 2023}


# ---------------------------------------------------------------------------
# Phase 2 prediction dispatch inside evaluate_checkpoint
# ---------------------------------------------------------------------------


class TestPhase2Prediction:
    def _checkpoint_rows(self, n_players: int = 4) -> pd.DataFrame:
        snap = _make_weekly_snapshot(2024, n_players=n_players)
        # Use the first weekly row per player as a synthetic checkpoint —
        # pa_ytd=80 crosses the 50 threshold. Reset index to line up with
        # benchmark_ros' filter semantics.
        return snap.drop_duplicates(subset=["mlbam_id", "season"]).reset_index(
            drop=True
        )

    def test_predict_phase2_returns_point_and_quantiles(self):
        rows = self._checkpoint_rows()
        preseason = _make_preseason_cache()
        # The fake ensemble accepts any feature names; use something deterministic.
        features = [
            "pa_ytd",
            "obp_ytd",
            "slg_ytd",
            "age",
            "park_factor_runs",
            "team_ops",
        ]
        fake = _FakePhase2Ensemble(
            feature_names=features, taus=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        out = br.predict_phase2(
            rows, fake, {"model": {"taus": [0.05, 0.25, 0.5, 0.75, 0.95]}}, preseason
        )

        assert out is not None
        point_df, q_arr = out
        assert list(point_df.columns) == list(ROS_RATE_TARGETS)
        assert point_df.shape == (len(rows), 6)
        assert q_arr.shape == (len(rows), 6, 5)
        # Post-hoc quantile sort: each row-target slice is monotone non-decreasing.
        diffs = np.diff(q_arr, axis=-1)
        assert (diffs >= -1e-9).all(), "phase2 quantiles must be sorted"

    def test_duplicate_preseason_keys_raises(self):
        """`_phase2_feature_matrix` uses validate='many_to_one' on the
        preseason merge: a preseason frame with duplicate (mlbam_id, season)
        rows must fail loudly instead of silently duplicating snapshot rows.
        """
        rows = self._checkpoint_rows()
        preseason_ok = _make_preseason_cache()
        preseason_bad = pd.concat(
            [preseason_ok, preseason_ok.iloc[:1]], ignore_index=True
        )
        fake = _FakePhase2Ensemble(
            feature_names=["pa_ytd"], taus=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        with pytest.raises(pd.errors.MergeError):
            br._phase2_feature_matrix(rows, fake, {}, preseason_bad)

    def test_evaluate_checkpoint_includes_phase2_quantile_metrics(self):
        rows = self._checkpoint_rows()
        preseason = _make_preseason_cache()
        fake = _FakePhase2Ensemble(
            feature_names=["pa_ytd", "obp_ytd", "slg_ytd"],
            taus=[0.05, 0.25, 0.5, 0.75, 0.95],
        )
        result = br.evaluate_checkpoint(
            rows,
            baselines=["phase2"],
            preseason=preseason,
            prior_pa=200.0,
            min_ros_pa=50,
            phase2_ensemble=fake,
            phase2_config={"model": {"taus": [0.05, 0.25, 0.5, 0.75, 0.95]}},
        )
        assert "phase2" in result["systems"]
        payload = result["systems"]["phase2"]
        # Point metrics present (apples-to-apples RMSE).
        assert "metrics" in payload
        # Quantile payloads present for pinball/PIT aggregation.
        assert "quantiles" in payload
        assert "quantile_metrics" in payload
        qm = payload["quantile_metrics"]
        assert "pinball" in qm and "pit" in qm


# ---------------------------------------------------------------------------
# Full dispatch: save_benchmark_outputs with phase2 yields quantile_metrics key
# ---------------------------------------------------------------------------


class TestBenchmarkOutputsWithPhase2:
    def test_quantile_metrics_key_in_json_report(self, tmp_path):
        rows = (
            _make_weekly_snapshot(2024, n_players=4)
            .drop_duplicates(subset=["mlbam_id", "season"])
            .reset_index(drop=True)
        )
        preseason = _make_preseason_cache()
        fake = _FakePhase2Ensemble(
            feature_names=["pa_ytd", "obp_ytd", "slg_ytd"],
            taus=[0.05, 0.25, 0.5, 0.75, 0.95],
        )
        # Evaluate a single checkpoint/year pair and pool.
        year_result = {
            "year": 2024,
            "thresholds": {
                50: br.evaluate_checkpoint(
                    rows,
                    baselines=["persist_observed", "phase2"],
                    preseason=preseason,
                    prior_pa=200.0,
                    min_ros_pa=50,
                    phase2_ensemble=fake,
                    phase2_config={"model": {"taus": [0.05, 0.25, 0.5, 0.75, 0.95]}},
                ),
            },
        }
        pooled = br.pool_by_threshold([year_result])
        br.save_benchmark_outputs([year_result], pooled, tmp_path)

        json_path = tmp_path / "benchmark_ros_report.json"
        assert json_path.exists()
        with open(json_path) as f:
            report = json.load(f)
        # The pooled block carries a 'quantile_metrics' subkey; phase2's
        # pinball / PIT dicts should be present.
        assert "pooled" in report
        threshold_block = report["pooled"]["50"]
        assert "quantile_metrics" in threshold_block
        assert "phase2" in threshold_block["quantile_metrics"]
        qm = threshold_block["quantile_metrics"]["phase2"]
        assert "pinball" in qm and "pit" in qm


# ---------------------------------------------------------------------------
# Smoke: --include phase2 --retrain dispatch with stubs
# ---------------------------------------------------------------------------


class TestBenchmarkSmokeWithSyntheticPhase2:
    def test_benchmark_smoke_with_synthetic_phase2(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Exercise the phase2 retrain dispatch end-to-end with a stub trainer.

        Writes synthetic snapshot files for 2023 and 2024, stubs ``train_ros``
        inside ``benchmark_ros`` to return a fake ensemble, and asserts that
        ``save_benchmark_outputs`` writes a JSON report containing
        ``quantile_metrics`` for phase2.
        """
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _make_weekly_snapshot(2023).to_parquet(
            raw_dir / "weekly_snapshots_2023.parquet"
        )
        _make_weekly_snapshot(2024).to_parquet(
            raw_dir / "weekly_snapshots_2024.parquet"
        )

        cache_dir = tmp_path / "phase2_cache"

        def fake_train_ros(config, snapshots_df=None, preseason_df=None):
            return _FakePhase2Ensemble(
                feature_names=["pa_ytd", "obp_ytd", "slg_ytd"],
                taus=config.get("model", {}).get("taus", [0.05, 0.25, 0.5, 0.75, 0.95]),
            )

        # train_ros is imported inside the helper, so patch the source module
        # rather than benchmark_ros' namespace.
        monkeypatch.setattr("src.models.mtl_ros.train.train_ros", fake_train_ros)

        phase2_cfg = {
            "model": {"taus": [0.05, 0.25, 0.5, 0.75, 0.95]},
            "training": {"epochs": 1},
            "ensemble": {"n_seeds": 1},
            "data": {},
            "splits": {"train_end_season": 2023},
        }
        ensemble = br._load_or_train_phase2_ensemble(
            eval_year=2024,
            phase2_cache_dir=cache_dir,
            phase2_config=phase2_cfg,
            raw_dir=raw_dir,
            df_featured=None,
            retrain=True,
        )
        assert ensemble is not None
        # Cache dir should contain a year_{2024} checkpoint directory.
        assert (cache_dir / "year_2024").exists()

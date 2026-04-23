"""Smoke tests for the Phase 2 ROS training pipeline.

These tests must pass without any real parquet files on disk: the synthetic
DataFrames built below carry every column ``train_ros`` needs, so the pipeline
exercises the full wiring (in-season features → preseason join → walk-forward
split → recency/sqrt(ros_pa) weights → ``MTLQuantileEnsembleForecaster.fit``)
without touching ``data/raw/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.mtl_ros.model import MTLQuantileEnsembleForecaster
from src.models.mtl_ros.train import train_ros


# ---------------------------------------------------------------------------
# Synthetic fixtures — no parquet files required.
# ---------------------------------------------------------------------------


def _make_synthetic_snapshots(
    seasons: tuple[int, ...] = (2020, 2021, 2022, 2023),
    players: tuple[int, ...] = (101, 102, 103, 104, 105),
    weeks_per_season: tuple[int, ...] = (20, 24, 28),
    seed: int = 0,
) -> pd.DataFrame:
    """~60 synthetic weekly snapshot rows with every column ``train_ros`` reads."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for season in seasons:
        for pid in players:
            for wk in weeks_per_season:
                pa_ytd = 60.0 + rng.uniform(0.0, 300.0)
                rows.append(
                    {
                        "mlbam_id": pid,
                        "season": season,
                        "iso_year": season,
                        "iso_week": int(wk),
                        # YTD counts needed by in-season features
                        "pa_ytd": pa_ytd,
                        "obp_ytd": float(rng.uniform(0.28, 0.40)),
                        "slg_ytd": float(rng.uniform(0.35, 0.50)),
                        "hr_per_pa_ytd": float(rng.uniform(0.01, 0.06)),
                        "r_per_pa_ytd": float(rng.uniform(0.08, 0.15)),
                        "rbi_per_pa_ytd": float(rng.uniform(0.08, 0.15)),
                        "sb_per_pa_ytd": float(rng.uniform(0.0, 0.03)),
                        "iso_ytd": float(rng.uniform(0.10, 0.25)),
                        "bb_rate_ytd": float(rng.uniform(0.06, 0.14)),
                        "k_rate_ytd": float(rng.uniform(0.15, 0.30)),
                        # Trailing 4-week counts so the in-season feature
                        # builder can derive the trail4w rates.
                        "trail4w_pa": float(rng.uniform(40.0, 100.0)),
                        "trail4w_h": float(rng.uniform(10.0, 30.0)),
                        "trail4w_bb": float(rng.uniform(2.0, 10.0)),
                        "trail4w_hbp": float(rng.uniform(0.0, 3.0)),
                        "trail4w_sf": float(rng.uniform(0.0, 2.0)),
                        "trail4w_ab": float(rng.uniform(30.0, 90.0)),
                        "trail4w_singles": float(rng.uniform(6.0, 20.0)),
                        "trail4w_doubles": float(rng.uniform(2.0, 8.0)),
                        "trail4w_triples": float(rng.uniform(0.0, 2.0)),
                        "trail4w_hr": float(rng.uniform(0.0, 5.0)),
                        "trail4w_r": float(rng.uniform(4.0, 15.0)),
                        "trail4w_rbi": float(rng.uniform(4.0, 15.0)),
                        "trail4w_sb": float(rng.uniform(0.0, 2.0)),
                        "trail4w_so": float(rng.uniform(6.0, 25.0)),
                        # ROS targets
                        "ros_pa": float(rng.uniform(50.0, 350.0)),
                        "ros_obp": float(rng.uniform(0.28, 0.40)),
                        "ros_slg": float(rng.uniform(0.35, 0.50)),
                        "ros_hr_per_pa": float(rng.uniform(0.01, 0.06)),
                        "ros_r_per_pa": float(rng.uniform(0.08, 0.15)),
                        "ros_rbi_per_pa": float(rng.uniform(0.08, 0.15)),
                        "ros_sb_per_pa": float(rng.uniform(0.0, 0.03)),
                    }
                )
    return pd.DataFrame(rows)


def _make_synthetic_preseason(
    snapshots: pd.DataFrame,
    seed: int = 1,
) -> pd.DataFrame:
    """One preseason row per (mlbam_id, season) with a handful of features."""
    rng = np.random.default_rng(seed)
    keys = snapshots[["mlbam_id", "season"]].drop_duplicates().reset_index(drop=True)
    keys["age"] = rng.uniform(22.0, 34.0, len(keys))
    keys["age_squared"] = keys["age"] ** 2
    keys["park_factor_runs"] = rng.uniform(95.0, 110.0, len(keys))
    keys["park_factor_hr"] = rng.uniform(90.0, 115.0, len(keys))
    keys["weighted_avg_obp"] = rng.uniform(0.28, 0.40, len(keys))
    keys["weighted_avg_slg"] = rng.uniform(0.35, 0.50, len(keys))
    keys["team_runs_per_game"] = rng.uniform(3.8, 5.5, len(keys))
    keys["team_ops"] = rng.uniform(0.680, 0.800, len(keys))
    return keys


def _tiny_config() -> dict:
    """Minimal config that keeps the ensemble fit under ~1s on CPU."""
    return {
        "model": {
            "n_quantiles": 5,
            "hidden_dims": [16, 8],
            "head_dim": 4,
            "dropouts": [0.1, 0.1],
            "two_stage": True,
            "use_residual": False,
            "speed_head_indices": [5],
            "speed_heads_receive_rates": False,
            "taus": [0.05, 0.25, 0.50, 0.75, 0.95],
        },
        "loss": {"pa_loss": "mse", "pa_weight": 1.0},
        "training": {
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "early_stopping_patience": 2,
            "recency_decay_lambda": 0.30,
            "min_ytd_pa": 50,
            "device": "cpu",
        },
        "ensemble": {"n_seeds": 1, "base_seed": 42},
        "data": {
            "preseason_feature_groups": {
                "age": True,
                "park_factors": True,
                "team_stats": True,
                "temporal": True,
            },
            "include_in_season_features": True,
        },
        "splits": {
            "train_end_season": 2022,
            "val_season": 2023,
            # No test split in the smoke test — train_ros shouldn't need one.
        },
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSmokeEndToEnd:
    def test_smoke_end_to_end(self, tmp_path):
        snapshots = _make_synthetic_snapshots()
        preseason = _make_synthetic_preseason(snapshots)
        cfg = _tiny_config()

        model = train_ros(cfg, snapshots_df=snapshots, preseason_df=preseason)
        assert isinstance(model, MTLQuantileEnsembleForecaster)

        # Predict on a tiny batch — we need the feature columns in the same
        # order the ensemble was trained with.
        X_probe = pd.DataFrame({f: [0.0, 0.0, 0.0] for f in model.feature_names_})
        preds = model.predict(X_probe)
        assert set(preds.keys()) == {"quantiles", "pa_remaining"}
        assert preds["quantiles"].shape == (3, 6, 5)
        assert preds["pa_remaining"].shape == (3, 1)
        assert np.all(np.isfinite(preds["quantiles"]))
        assert np.all(np.isfinite(preds["pa_remaining"]))

        # Save/load roundtrip.
        save_dir = tmp_path / "ros_smoke"
        model.save(save_dir)
        loaded = MTLQuantileEnsembleForecaster.load(save_dir)
        preds2 = loaded.predict(X_probe)
        np.testing.assert_allclose(preds["quantiles"], preds2["quantiles"], atol=1e-4)
        np.testing.assert_allclose(
            preds["pa_remaining"], preds2["pa_remaining"], atol=1e-3
        )


class TestSmokeMultiSeed:
    def test_smoke_multi_seed(self):
        snapshots = _make_synthetic_snapshots()
        preseason = _make_synthetic_preseason(snapshots)
        cfg = _tiny_config()
        cfg["ensemble"]["n_seeds"] = 2

        model = train_ros(cfg, snapshots_df=snapshots, preseason_df=preseason)
        assert len(model.forecasters_) == 2
        # Distinct seeds → distinct (seed-dependent) weights on each member.
        assert model.forecasters_[0].seed != model.forecasters_[1].seed

        X_probe = pd.DataFrame({f: [0.0] for f in model.feature_names_})
        preds = model.predict(X_probe)
        assert preds["quantiles"].shape == (1, 6, 5)


class TestConfigLoad:
    def test_default_config_is_valid_yaml(self):
        """configs/mtl_ros.yaml must parse and define every required key."""
        cfg_path = Path("configs/mtl_ros.yaml")
        assert cfg_path.exists(), (
            "configs/mtl_ros.yaml is missing; Task 4 should create it."
        )
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        for section in (
            "model",
            "loss",
            "training",
            "ensemble",
            "data",
            "splits",
            "output",
        ):
            assert section in cfg, f"mtl_ros.yaml missing section {section!r}"
        assert "n_quantiles" in cfg["model"]
        assert "taus" in cfg["model"]
        assert "train_end_season" in cfg["splits"]
        assert "val_season" in cfg["splits"]
        assert "test_season" in cfg["splits"]


class TestFeatureSelection:
    def test_feature_selection_excludes_missing_warns(self, caplog):
        """Dropping an expected preseason column should warn but still train."""
        snapshots = _make_synthetic_snapshots()
        preseason = _make_synthetic_preseason(snapshots).drop(
            columns=["park_factor_hr"]
        )
        cfg = _tiny_config()

        with caplog.at_level("WARNING"):
            model = train_ros(cfg, snapshots_df=snapshots, preseason_df=preseason)
        assert isinstance(model, MTLQuantileEnsembleForecaster)
        # The missing feature should not silently survive in the feature list.
        assert "park_factor_hr" not in model.feature_names_
        # Some warning should have mentioned the missing column.
        warn_text = "\n".join(r.message for r in caplog.records)
        assert "park_factor_hr" in warn_text


class TestEmptyTrainFrameRaises:
    """When every train row has a NaN ROS target (e.g. bad snapshot data),
    ``train_ros`` must raise a clear ValueError instead of passing an
    empty array into ``StandardScaler.fit_transform``.
    """

    def test_all_nan_targets_raises(self):
        snapshots = _make_synthetic_snapshots()
        # Wipe every row's ROS rate target so the nan-drop removes them all.
        for col in (
            "ros_obp",
            "ros_slg",
            "ros_hr_per_pa",
            "ros_r_per_pa",
            "ros_rbi_per_pa",
            "ros_sb_per_pa",
        ):
            snapshots[col] = np.nan
        preseason = _make_synthetic_preseason(snapshots)
        cfg = _tiny_config()
        with pytest.raises(ValueError, match="No usable training rows"):
            train_ros(cfg, snapshots_df=snapshots, preseason_df=preseason)


class TestInsufficientTrainBatchRaises:
    """With ``drop_last=True`` on the training DataLoader (F8 fix), a
    train frame smaller than ``batch_size`` would silently produce an
    empty loader → unfit model. ``train_ros`` must fail loudly.
    """

    def test_train_rows_below_batch_size_raises(self):
        snapshots = _make_synthetic_snapshots(
            seasons=(2020, 2021, 2022),
            players=(101, 102),
            weeks_per_season=(20,),
        )
        preseason = _make_synthetic_preseason(snapshots)
        cfg = _tiny_config()
        # Train split = seasons ≤ 2022 → 4 rows. batch_size = 64 forces <.
        cfg["training"]["batch_size"] = 64
        with pytest.raises(ValueError, match="Insufficient training rows"):
            train_ros(cfg, snapshots_df=snapshots, preseason_df=preseason)


class TestCLISmokeFlag:
    def test_cli_smoke_flag_runs_to_success(self, tmp_path, monkeypatch):
        """`main(--smoke)` builds a tiny model end-to-end and exits 0."""
        from src.models.mtl_ros import train as train_module

        out_dir = tmp_path / "ros_cli_smoke"
        # Keep tests hermetic by pointing the output at tmp_path.
        rc = train_module.main(
            argv=["--smoke", "--out", str(out_dir), "--device", "cpu"]
        )
        assert rc == 0
        # The --smoke path should have left a saved ensemble behind.
        assert (out_dir / "ensemble_meta.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

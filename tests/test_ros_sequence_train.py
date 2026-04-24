"""Smoke tests for Phase 3 ROSSequenceForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.mtl_ros.model import MTLQuantileForecaster
from src.models.ros.model import ROSSequenceForecaster


def _base_forecaster(n_features: int = 6) -> MTLQuantileForecaster:
    rng = np.random.default_rng(4)
    X = pd.DataFrame(
        rng.normal(size=(24, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.DataFrame(
        {
            "ros_obp": rng.normal(0.33, 0.02, len(X)),
            "ros_slg": rng.normal(0.43, 0.04, len(X)),
            "ros_hr_per_pa": rng.normal(0.03, 0.005, len(X)),
            "ros_r_per_pa": rng.normal(0.12, 0.01, len(X)),
            "ros_rbi_per_pa": rng.normal(0.12, 0.01, len(X)),
            "ros_sb_per_pa": rng.normal(0.02, 0.005, len(X)),
        }
    )
    pa = rng.normal(260, 40, len(X))
    cfg = {
        "seed": 4,
        "model": {
            "hidden_dims": [12, 8],
            "head_dim": 4,
            "dropouts": [0.0, 0.0],
            "n_quantiles": 5,
            "taus": [0.05, 0.25, 0.5, 0.75, 0.95],
            "two_stage": True,
            "speed_head_indices": [5],
        },
        "loss": {"pa_loss": "mse", "pa_weight": 0.1},
        "training": {"batch_size": 8, "epochs": 1, "learning_rate": 0.01},
    }
    return MTLQuantileForecaster(cfg).fit(X, y, pa_target=pa)


def _snapshots(n_rows: int = 12) -> pd.DataFrame:
    rows: list[dict] = []
    for player in (100, 101, 102):
        for i, week in enumerate((14, 15, 16, 17)):
            pa_ytd = 60.0 + i * 35.0
            rows.append(
                {
                    "mlbam_id": player,
                    "season": 2025,
                    "iso_year": 2025,
                    "iso_week": week,
                    "pa_week": 35.0,
                    "ab_week": 30.0,
                    "h_week": 9.0,
                    "singles_week": 6.0,
                    "doubles_week": 2.0,
                    "triples_week": 0.0,
                    "hr_week": 1.0,
                    "r_week": 4.0,
                    "rbi_week": 4.0,
                    "bb_week": 4.0,
                    "so_week": 7.0,
                    "hbp_week": 1.0,
                    "sf_week": 0.0,
                    "sb_week": 1.0,
                    "cs_week": 0.0,
                    "pa_ytd": pa_ytd,
                    "ros_pa": 360.0 - pa_ytd,
                    "ros_obp": 0.330 + 0.001 * i,
                    "ros_slg": 0.430 + 0.002 * i,
                    "ros_hr_per_pa": 0.030,
                    "ros_r_per_pa": 0.120,
                    "ros_rbi_per_pa": 0.125,
                    "ros_sb_per_pa": 0.020,
                }
            )
    return pd.DataFrame(rows).iloc[:n_rows].reset_index(drop=True)


def test_sequence_forecaster_fit_predict_save_load(tmp_path) -> None:
    base = _base_forecaster()
    rows = _snapshots()
    phase2_x = pd.DataFrame(
        np.zeros((len(rows), len(base.feature_names_))),
        columns=base.feature_names_,
    )
    targets = rows[
        [
            "ros_obp",
            "ros_slg",
            "ros_hr_per_pa",
            "ros_r_per_pa",
            "ros_rbi_per_pa",
            "ros_sb_per_pa",
        ]
    ]
    cfg = {
        "seed": 7,
        "model": {
            "encoder_dim": 4,
            "gru_hidden_dim": 8,
            "dropout": 0.0,
            "max_seq_len": 8,
        },
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "early_stopping_patience": 2,
            "device": "cpu",
        },
        "loss": {"pa_weight": 0.1},
    }

    model = ROSSequenceForecaster(base, cfg).fit(
        rows,
        phase2_x,
        targets,
        pa_target=rows["ros_pa"],
    )
    preds = model.predict(rows, phase2_x)
    assert preds["quantiles"].shape == (len(rows), 6, 5)
    assert preds["pa_remaining"].shape == (len(rows), 1)
    assert np.isfinite(preds["quantiles"]).all()

    save_dir = tmp_path / "phase3"
    model.save(save_dir)
    loaded = ROSSequenceForecaster.load(save_dir)
    preds_loaded = loaded.predict(rows, phase2_x)
    np.testing.assert_allclose(preds["quantiles"], preds_loaded["quantiles"], atol=1e-4)
    np.testing.assert_allclose(
        preds["pa_remaining"], preds_loaded["pa_remaining"], atol=1e-3
    )

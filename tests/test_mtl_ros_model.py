"""Tests for the MTL ROS quantile network and forecaster.

Covers:
- MTLQuantileNetwork forward output shapes (quantile + PA-remaining heads)
- Two-stage rate→count wiring in the ROS variant
- Forecaster fit/predict roundtrip on tiny synthetic data
- Forecaster save/load roundtrip preserving predictions
- Ensemble per-quantile mean across seeds
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import torch

from src.models.mtl_ros.model import (
    MTLQuantileEnsembleForecaster,
    MTLQuantileForecaster,
    MTLQuantileNetwork,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_training_data(
    n_samples: int = 30,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Synthetic (X, y, pa_target) for tiny-scale forecaster tests."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n_samples, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.DataFrame(
        {
            "target_obp": rng.normal(0.32, 0.04, n_samples).clip(0.2, 0.5),
            "target_slg": rng.normal(0.42, 0.07, n_samples).clip(0.25, 0.7),
            "target_hr_per_pa": rng.normal(0.03, 0.015, n_samples).clip(0.0, 0.1),
            "target_r_per_pa": rng.normal(0.11, 0.03, n_samples).clip(0.02, 0.2),
            "target_rbi_per_pa": rng.normal(0.11, 0.03, n_samples).clip(0.02, 0.2),
            "target_sb_per_pa": rng.normal(0.015, 0.02, n_samples).clip(0.0, 0.15),
        }
    )
    pa_target = rng.normal(300, 80, n_samples).clip(50, 700)
    return X, y, pa_target


def _make_small_config(epochs: int = 3) -> dict:
    """Fast config for tests — tiny backbone, few epochs."""
    return {
        "seed": 42,
        "model": {
            "hidden_dims": [16, 8],
            "head_dim": 4,
            "dropouts": [0.1, 0.1],
            "n_quantiles": 5,
            "taus": [0.05, 0.25, 0.5, 0.75, 0.95],
            "two_stage": True,
            "speed_head_indices": [5],
            "speed_heads_receive_rates": False,
        },
        "loss": {"pa_loss": "mse", "pa_weight": 1.0},
        "training": {
            "batch_size": 8,
            "epochs": epochs,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "early_stopping_patience": 2,
            "recency_decay_lambda": 0.0,
        },
        "ensemble": {"n_seeds": 2, "base_seed": 42},
    }


# ---------------------------------------------------------------------------
# TestMTLQuantileNetwork
# ---------------------------------------------------------------------------


class TestMTLQuantileNetwork:
    """Tests for the MTLQuantileNetwork nn.Module."""

    def test_forward_output_shapes(self):
        """Dict output: quantiles (B, T, Q), pa_remaining (B, 1)."""
        net = MTLQuantileNetwork(n_features=50, n_targets=6, n_quantiles=5)
        net.eval()  # avoid BatchNorm1d issues on batch>1
        x = torch.randn(8, 50)
        out = net(x)

        assert isinstance(out, dict)
        assert "quantiles" in out
        assert "pa_remaining" in out
        assert out["quantiles"].shape == (8, 6, 5)
        assert out["pa_remaining"].shape == (8, 1)

    def test_custom_quantile_count(self):
        """n_quantiles=3 is honoured."""
        net = MTLQuantileNetwork(
            n_features=20,
            n_targets=6,
            n_quantiles=3,
            taus=(0.1, 0.5, 0.9),
        )
        net.eval()
        x = torch.randn(4, 20)
        out = net(x)
        assert out["quantiles"].shape == (4, 6, 3)

    def test_taus_stored_as_buffer(self):
        """taus is a non-trainable buffer, not a parameter."""
        net = MTLQuantileNetwork(
            n_features=10,
            n_targets=6,
            n_quantiles=5,
            taus=(0.05, 0.25, 0.5, 0.75, 0.95),
        )
        # Must be registered as a buffer, not a parameter
        buffer_names = dict(net.named_buffers())
        assert "taus" in buffer_names
        assert buffer_names["taus"].requires_grad is False
        torch.testing.assert_close(
            buffer_names["taus"],
            torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]),
        )

    def test_two_stage_param_count_differs(self):
        """two_stage count heads take +2 inputs (from rate medians) → more params."""
        net_one = MTLQuantileNetwork(
            n_features=10,
            n_targets=6,
            two_stage=False,
        )
        net_two = MTLQuantileNetwork(
            n_features=10,
            n_targets=6,
            two_stage=True,
            speed_head_indices=(),  # all count heads receive rates
        )
        p_one = sum(p.numel() for p in net_one.parameters())
        p_two = sum(p.numel() for p in net_two.parameters())
        assert p_two > p_one

    def test_speed_head_skips_rate_input_by_default(self):
        """With speed_head_indices=(5,) and speed_heads_receive_rates=False,
        the speed head (index 5) should use pure backbone output."""
        net = MTLQuantileNetwork(
            n_features=10,
            n_targets=6,
            n_quantiles=5,
            two_stage=True,
            speed_head_indices=(5,),
            speed_heads_receive_rates=False,
        )
        net.eval()
        # Inspect the in_features of the first Linear in the speed head
        speed_head = net.heads[5]
        first_linear = speed_head[0]  # nn.Sequential: Linear, ReLU, Linear
        backbone_out = 64 if net.hidden_dims[-1] == 64 else net.hidden_dims[-1]
        assert first_linear.in_features == backbone_out

    def test_pa_head_present(self):
        """PA-remaining head maps backbone → (B, 1)."""
        net = MTLQuantileNetwork(n_features=10, n_targets=6)
        assert hasattr(net, "pa_head")
        net.eval()
        out = net(torch.randn(3, 10))
        assert out["pa_remaining"].shape == (3, 1)


# ---------------------------------------------------------------------------
# TestMTLQuantileForecaster
# ---------------------------------------------------------------------------


class TestMTLQuantileForecaster:
    """Sklearn-style forecaster wrapper."""

    def test_fit_predict_shapes(self):
        X, y, pa_target = _make_training_data()
        cfg = _make_small_config(epochs=3)
        model = MTLQuantileForecaster(cfg)
        model.fit(X, y, pa_target=pa_target)

        preds = model.predict(X)
        assert isinstance(preds, dict)
        assert "quantiles" in preds
        assert "pa_remaining" in preds
        assert preds["quantiles"].shape == (len(X), 6, 5)
        assert preds["pa_remaining"].shape == (len(X), 1)
        assert np.all(np.isfinite(preds["quantiles"]))
        assert np.all(np.isfinite(preds["pa_remaining"]))

    def test_pa_target_is_scaled_in_training(self):
        """PA target MUST be z-scored in training so the MSE loss sits on the
        same scale as pinball rate losses. Raw PA magnitudes (~100-700)
        would otherwise make pa_weight=1.0 dominate the combined loss.
        """
        X, y, pa_target = _make_training_data()
        cfg = _make_small_config(epochs=2)
        model = MTLQuantileForecaster(cfg)
        model.fit(X, y, pa_target=pa_target)
        assert model.pa_scaler_ is not None
        # Scaler fitted on PA should recover the training distribution's
        # mean within a reasonable tolerance.
        assert np.isclose(model.pa_scaler_.mean_[0], pa_target.mean(), rtol=1e-5)
        assert model.pa_scaler_.scale_[0] > 0

    def test_pa_prediction_in_raw_units(self):
        """After training on raw PA targets, ``predict`` must return PA
        predictions back in the raw PA unit space (not z-scored).
        """
        X, y, pa_target = _make_training_data(n_samples=60)
        cfg = _make_small_config(epochs=5)
        model = MTLQuantileForecaster(cfg)
        model.fit(X, y, pa_target=pa_target)

        preds = model.predict(X)
        pa_pred = preds["pa_remaining"].flatten()
        # Should be in roughly the same magnitude as the training PA
        # distribution — the model may not hit exact numbers but should
        # land in the raw PA scale [50, 700] rather than z-space [-3, 3].
        assert pa_pred.mean() > 50
        assert pa_pred.mean() < 700

    def test_drop_last_avoids_batchnorm_batch_of_1_crash(self):
        """Training with ``len(dataset) % batch_size == 1`` must not crash
        BatchNorm1d. The forecaster's DataLoader uses ``drop_last=True`` to
        skip the tail-1 batch.
        """
        # 9 samples with batch_size=8 → last batch has 1 sample.
        X, y, pa_target = _make_training_data(n_samples=9)
        cfg = _make_small_config(epochs=2)
        cfg["training"]["batch_size"] = 8
        model = MTLQuantileForecaster(cfg)
        # Previously this would raise: "Expected more than 1 value per channel
        # when training, got input size torch.Size([1, N])".
        model.fit(X, y, pa_target=pa_target)
        assert model.is_fitted_

    def test_device_config_threads_through(self):
        """The model must honour ``training.device`` — network lives on the
        configured device, predict works without errors.
        """
        X, y, pa_target = _make_training_data()
        cfg = _make_small_config(epochs=2)
        cfg["training"]["device"] = "cpu"
        model = MTLQuantileForecaster(cfg)
        model.fit(X, y, pa_target=pa_target)
        # Network parameters should all live on the configured device.
        param_devices = {str(p.device) for p in model.network_.parameters()}
        assert param_devices == {"cpu"}
        preds = model.predict(X)
        assert np.all(np.isfinite(preds["quantiles"]))

    def test_fit_predict_with_eval_set(self):
        X, y, pa_target = _make_training_data(n_samples=40)
        X_val, y_val, pa_val = _make_training_data(n_samples=10, seed=7)
        cfg = _make_small_config(epochs=3)
        model = MTLQuantileForecaster(cfg)
        model.fit(
            X,
            y,
            pa_target=pa_target,
            eval_set=(X_val, y_val, pa_val),
        )
        preds = model.predict(X_val)
        assert preds["quantiles"].shape == (10, 6, 5)

    def test_save_load_roundtrip(self, tmp_path):
        X, y, pa_target = _make_training_data()
        cfg = _make_small_config(epochs=3)
        model = MTLQuantileForecaster(cfg)
        model.fit(X, y, pa_target=pa_target)
        p1 = model.predict(X)

        save_dir = tmp_path / "mtl_ros_model"
        model.save(save_dir)

        loaded = MTLQuantileForecaster.load(save_dir)
        p2 = loaded.predict(X)

        np.testing.assert_allclose(
            p1["quantiles"],
            p2["quantiles"],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            p1["pa_remaining"],
            p2["pa_remaining"],
            atol=1e-4,
        )

    def test_unfitted_predict_raises(self):
        model = MTLQuantileForecaster(_make_small_config())
        import pytest

        with pytest.raises(RuntimeError):
            model.predict(pd.DataFrame(np.zeros((3, 5))))

    def test_forecaster_predict_returns_monotonic_quantiles(self):
        """Every (row, target) quantile vector must be sorted along tau.

        Pinball-loss training on a tiny backbone does not guarantee
        tau-ordered outputs on unseen rows, so ``predict`` enforces
        monotonicity before returning. Regression guard: without the
        post-hoc sort we empirically see hundreds of negative per-tau
        diffs on this synthetic smoke model.
        """
        X, y, pa_target = _make_training_data(n_samples=40)
        cfg = _make_small_config(epochs=2)
        model = MTLQuantileForecaster(cfg)
        model.fit(X, y, pa_target=pa_target)

        preds = model.predict(X)
        diffs = np.diff(preds["quantiles"], axis=-1)
        # All per-tau gaps must be >= 0 (non-decreasing across tau).
        assert np.all(diffs >= 0), (
            f"Quantile crossings detected: min diff = {diffs.min()}"
        )


# ---------------------------------------------------------------------------
# TestMTLQuantileEnsembleForecaster
# ---------------------------------------------------------------------------


class TestMTLQuantileEnsembleForecaster:
    """Multi-seed ensemble wrapper."""

    def test_fit_predict_shape(self):
        X, y, pa_target = _make_training_data()
        cfg = _make_small_config(epochs=2)
        ens = MTLQuantileEnsembleForecaster(cfg)
        ens.fit(X, y, pa_target=pa_target)
        assert len(ens.forecasters_) == cfg["ensemble"]["n_seeds"]

        preds = ens.predict(X)
        assert preds["quantiles"].shape == (len(X), 6, 5)
        assert preds["pa_remaining"].shape == (len(X), 1)

    def test_per_quantile_mean_aggregation(self):
        """Ensemble prediction == element-wise mean of member predictions."""
        # Build a zero-seed ensemble, then swap in two stub forecasters
        # whose predict() returns known arrays.
        cfg = _make_small_config(epochs=1)
        ens = MTLQuantileEnsembleForecaster(cfg)

        # Use tau-sorted member arrays so the ensemble-side sort is a no-op
        # here. Monotonicity under member disagreement is covered by the
        # dedicated test below.
        tau_row = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
        member_a = MagicMock()
        member_a.predict.return_value = {
            "quantiles": np.broadcast_to(tau_row, (3, 6, 5)) * 2.0,
            "pa_remaining": np.ones((3, 1)) * 100.0,
        }
        member_b = MagicMock()
        member_b.predict.return_value = {
            "quantiles": np.broadcast_to(tau_row, (3, 6, 5)) * 4.0,
            "pa_remaining": np.ones((3, 1)) * 200.0,
        }
        ens.forecasters_ = [member_a, member_b]
        ens.is_fitted_ = True
        ens.feature_names_ = [f"f{i}" for i in range(5)]
        ens.target_names_ = [
            "target_obp",
            "target_slg",
            "target_hr_per_pa",
            "target_r_per_pa",
            "target_rbi_per_pa",
            "target_sb_per_pa",
        ]

        X = pd.DataFrame(np.zeros((3, 5)))
        preds = ens.predict(X)
        # Mean of 2*tau_row and 4*tau_row is 3*tau_row — per-quantile, per-task, per-row.
        np.testing.assert_allclose(
            preds["quantiles"],
            np.broadcast_to(tau_row * 3.0, (3, 6, 5)),
        )
        np.testing.assert_allclose(preds["pa_remaining"], np.ones((3, 1)) * 150.0)

    def test_ensemble_predict_returns_monotonic_quantiles(self):
        """Ensemble sort fixes crossings introduced by member disagreement.

        Two stub members emit tau-sorted arrays, but they disagree on the
        per-tau ranking so the element-wise mean is NOT sorted. The
        ensemble's predict() must return a sorted result anyway.
        """
        cfg = _make_small_config(epochs=1)
        ens = MTLQuantileEnsembleForecaster(cfg)

        # Both members emit individually-sorted quantiles but with opposite
        # tails: averaging produces a crossing at the middle.
        member_a = MagicMock()
        member_a.predict.return_value = {
            "quantiles": np.broadcast_to(
                np.array([0.1, 0.2, 0.3, 0.9, 1.0]), (2, 6, 5)
            ).copy(),
            "pa_remaining": np.zeros((2, 1)),
        }
        member_b = MagicMock()
        member_b.predict.return_value = {
            "quantiles": np.broadcast_to(
                np.array([0.0, 0.05, 0.8, 0.85, 0.9]), (2, 6, 5)
            ).copy(),
            "pa_remaining": np.zeros((2, 1)),
        }
        # Unsorted mean would be [0.05, 0.125, 0.55, 0.875, 0.95] — already
        # sorted. Craft a crossing by making member_b's q_0.75 < q_0.50 after
        # averaging: swap member_b's 2nd and 3rd entries.
        member_b.predict.return_value["quantiles"] = np.broadcast_to(
            np.array([0.0, 0.8, 0.05, 0.85, 0.9]), (2, 6, 5)
        ).copy()
        # Mean: [0.05, 0.5, 0.175, 0.875, 0.95] — crosses between idx 1 and 2.
        ens.forecasters_ = [member_a, member_b]
        ens.is_fitted_ = True
        ens.feature_names_ = [f"f{i}" for i in range(5)]
        ens.target_names_ = ["t"] * 6

        X = pd.DataFrame(np.zeros((2, 5)))
        preds = ens.predict(X)
        diffs = np.diff(preds["quantiles"], axis=-1)
        assert np.all(diffs >= 0), (
            f"Ensemble quantile crossings detected: min diff = {diffs.min()}"
        )

    def test_save_load_roundtrip(self, tmp_path):
        X, y, pa_target = _make_training_data(n_samples=20)
        cfg = _make_small_config(epochs=2)
        ens = MTLQuantileEnsembleForecaster(cfg)
        ens.fit(X, y, pa_target=pa_target)
        p1 = ens.predict(X)

        save_dir = tmp_path / "mtl_ros_ensemble"
        ens.save(save_dir)
        loaded = MTLQuantileEnsembleForecaster.load(save_dir)
        p2 = loaded.predict(X)

        np.testing.assert_allclose(p1["quantiles"], p2["quantiles"], atol=1e-4)
        np.testing.assert_allclose(p1["pa_remaining"], p2["pa_remaining"], atol=1e-3)

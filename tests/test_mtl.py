"""Tests for the MTL neural network

Covers:
- MTLNetwork forward pass shapes and output structure
- MultiTaskLoss computation and learned parameters
- BatterDataset construction and NaN handling
- MTLForecaster fit/predict on synthetic data
- Output shape and type correctness
- NaN handling in features (fill with 0 after scaling)
- Early stopping with eval_set
- Model save/load roundtrip (state_dict based)
- Learned task weight extraction
- Evaluation report generation with MTL
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.eval.metrics import compare_to_baseline, compute_metrics, compute_naive_baseline
from src.eval.report import print_report, save_report
from src.features.registry import TARGET_COLUMNS, TARGET_DISPLAY, TARGET_STATS
from src.models.mtl.dataset import BatterDataset
from src.models.mtl.loss import HuberMultiTaskLoss, MultiTaskLoss
from src.models.mtl.model import (
    GatedTaskHead,
    MTLEnsembleForecaster,
    MTLForecaster,
    MTLNetwork,
    ResidualBlock,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Realistic target value ranges for synthetic data
_TARGET_RANGES: dict[str, tuple[float, float, float]] = {
    # (mean, std, clip_min)
    "target_obp": (0.320, 0.035, 0.200),
    "target_slg": (0.420, 0.070, 0.250),
    "target_hr": (18.0, 10.0, 0.0),
    "target_r": (65.0, 20.0, 10.0),
    "target_rbi": (60.0, 22.0, 5.0),
    "target_sb": (8.0, 7.0, 0.0),
}


def _make_training_data(
    n_samples: int = 100,
    n_features: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic training data with realistic target ranges."""
    rng = np.random.RandomState(seed)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=feature_names)

    y_data: dict[str, np.ndarray] = {}
    for name, (mean, std, clip_min) in _TARGET_RANGES.items():
        vals = rng.normal(mean, std, n_samples)
        y_data[name] = np.clip(vals, clip_min, None)

    y = pd.DataFrame(y_data)
    return X, y


def _make_small_config(epochs: int = 5) -> dict:
    """Config for fast unit tests (tiny architecture)."""
    return {
        "model": {
            "hidden_dims": [16, 8],
            "head_dim": 4,
            "dropouts": [0.1, 0.1],
            "batch_size": 16,
            "epochs": epochs,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "early_stopping_patience": 3,
            "lr_scheduler": {
                "patience": 2,
                "factor": 0.5,
                "min_lr": 1e-6,
            },
        },
        "seed": 42,
    }


# ---------------------------------------------------------------------------
# TestMTLNetwork
# ---------------------------------------------------------------------------


class TestMTLNetwork:
    """Tests for the pure PyTorch nn.Module."""

    def test_forward_output_type(self):
        net = MTLNetwork(n_features=10, n_targets=6)
        x = torch.randn(5, 10)
        out = net(x)
        assert isinstance(out, list)
        assert len(out) == 6

    def test_forward_output_shapes(self):
        net = MTLNetwork(n_features=10, n_targets=6)
        x = torch.randn(5, 10)
        out = net(x)
        for t in out:
            assert t.shape == (5, 1)

    def test_custom_architecture(self):
        net = MTLNetwork(
            n_features=20,
            n_targets=3,
            hidden_dims=[64, 32],
            head_dim=8,
            dropouts=[0.1, 0.1],
        )
        x = torch.randn(4, 20)
        out = net(x)
        assert len(out) == 3
        for t in out:
            assert t.shape == (4, 1)

    def test_batch_norm_present(self):
        net = MTLNetwork(n_features=10)
        assert hasattr(net, "batch_norm")
        assert isinstance(net.batch_norm, nn.BatchNorm1d)

    def test_has_learnable_parameters(self):
        net = MTLNetwork(n_features=10)
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        assert n_params > 0

    def test_single_target(self):
        net = MTLNetwork(n_features=5, n_targets=1)
        x = torch.randn(3, 5)
        out = net(x)
        assert len(out) == 1
        assert out[0].shape == (3, 1)

    def test_mismatched_dims_raises(self):
        with pytest.raises(ValueError, match="same length"):
            MTLNetwork(n_features=10, hidden_dims=[64, 32], dropouts=[0.1])


# ---------------------------------------------------------------------------
# TestMultiTaskLoss
# ---------------------------------------------------------------------------


class TestMultiTaskLoss:
    """Tests for the uncertainty-weighted loss."""

    def test_output_type(self):
        loss_fn = MultiTaskLoss(n_tasks=6)
        preds = [torch.randn(4, 1) for _ in range(6)]
        targets = torch.randn(4, 6)
        loss, details = loss_fn(preds, targets)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(details, dict)

    def test_loss_is_scalar(self):
        loss_fn = MultiTaskLoss(n_tasks=3)
        preds = [torch.randn(4, 1) for _ in range(3)]
        targets = torch.randn(4, 3)
        loss, _ = loss_fn(preds, targets)
        assert loss.dim() == 0  # scalar

    def test_loss_is_positive(self):
        loss_fn = MultiTaskLoss(n_tasks=6)
        preds = [torch.randn(4, 1) for _ in range(6)]
        targets = torch.randn(4, 6)
        loss, _ = loss_fn(preds, targets)
        assert loss.item() > 0

    def test_log_vars_are_parameters(self):
        loss_fn = MultiTaskLoss(n_tasks=6)
        assert isinstance(loss_fn.log_vars, nn.Parameter)
        param_list = list(loss_fn.parameters())
        assert len(param_list) == 1
        assert param_list[0] is loss_fn.log_vars

    def test_details_dict_keys(self):
        loss_fn = MultiTaskLoss(n_tasks=3)
        preds = [torch.randn(4, 1) for _ in range(3)]
        targets = torch.randn(4, 3)
        _, details = loss_fn(preds, targets)
        assert "total" in details
        for t in range(3):
            assert f"task_{t}_mse" in details
            assert f"task_{t}_weight" in details

    def test_initial_task_weights_are_one(self):
        """With log_var=0, precision = exp(0) = 1."""
        loss_fn = MultiTaskLoss(n_tasks=6)
        weights = loss_fn.get_task_weights()
        np.testing.assert_allclose(weights, np.ones(6), atol=1e-6)


# ---------------------------------------------------------------------------
# TestBatterDataset
# ---------------------------------------------------------------------------


class TestBatterDataset:
    """Tests for the PyTorch Dataset."""

    def test_length(self):
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 6)
        ds = BatterDataset(X, y)
        assert len(ds) == 50

    def test_getitem_with_targets(self):
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 3)
        ds = BatterDataset(X, y)
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 3  # (features, targets, weight)
        assert item[0].shape == (5,)
        assert item[1].shape == (3,)
        assert item[2].item() == pytest.approx(1.0)  # default weight

    def test_getitem_without_targets(self):
        X = np.random.randn(10, 5)
        ds = BatterDataset(X, y=None)
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5,)

    def test_nan_replaced_with_zero(self):
        X = np.array([[1.0, np.nan, 3.0], [np.nan, 2.0, np.nan]])
        ds = BatterDataset(X)
        assert ds.X[0, 1].item() == 0.0
        assert ds.X[1, 0].item() == 0.0
        assert ds.X[1, 2].item() == 0.0

    def test_dtype_is_float32(self):
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 3)
        ds = BatterDataset(X, y)
        assert ds.X.dtype == torch.float32
        assert ds.y.dtype == torch.float32


# ---------------------------------------------------------------------------
# TestMTLForecaster
# ---------------------------------------------------------------------------


class TestMTLForecaster:
    """Core fit/predict tests for MTLForecaster."""

    @pytest.fixture()
    def training_data(self):
        return _make_training_data(n_samples=100, n_features=10)

    @pytest.fixture()
    def default_config(self):
        return _make_small_config()

    def test_fit_returns_self(self, training_data, default_config):
        X, y = training_data
        model = MTLForecaster(default_config)
        result = model.fit(X, y)
        assert result is model

    def test_is_fitted_after_fit(self, training_data, default_config):
        X, y = training_data
        model = MTLForecaster(default_config)
        assert not model.is_fitted_
        model.fit(X, y)
        assert model.is_fitted_

    def test_predict_shape(self, training_data, default_config):
        X, y = training_data
        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100, 6)

    def test_predict_column_names(self, training_data, default_config):
        X, y = training_data
        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert list(preds.columns) == TARGET_COLUMNS

    def test_predict_no_nan(self, training_data, default_config):
        X, y = training_data
        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.isna().sum().sum() == 0

    def test_predict_unfitted_raises(self, training_data):
        X, _ = training_data
        model = MTLForecaster()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_fit_with_numpy(self, default_config):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 5)
        y = rng.randn(50, 6)
        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50, 6)

    def test_predictions_are_reasonable(self, default_config):
        """Train longer and check predictions are in a reasonable range."""
        config = _make_small_config(epochs=30)
        X, y = _make_training_data(n_samples=200, n_features=10)
        model = MTLForecaster(config)
        model.fit(X, y)
        preds = model.predict(X)
        # Training RMSE should be within reasonable bounds
        for col in y.columns:
            residuals = preds[col].values - y[col].values
            train_rmse = np.sqrt(np.mean(residuals**2))
            target_std = y[col].std()
            # Training RMSE should be less than 2x the target std
            assert train_rmse < 2 * target_std, (
                f"{col}: train RMSE {train_rmse:.4f} > 2x std {target_std:.4f}"
            )

    def test_default_config(self, training_data):
        X, y = training_data
        model = MTLForecaster()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100, 6)

    def test_handles_nan_in_features(self, default_config):
        """NaN features should be handled via scaling + fill with 0."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(80, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame(
            {name: rng.normal(m, s, 80).clip(c) for name, (m, s, c) in _TARGET_RANGES.items()}
        )

        # Inject 20% NaN into features
        mask = rng.random((80, 5)) < 0.2
        X = X.mask(pd.DataFrame(mask, columns=X.columns))

        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.isna().sum().sum() == 0

    def test_fit_with_single_target(self, default_config):
        """Should work with a single target column."""
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(50, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({"target_hr": rng.normal(18, 10, 50).clip(0)})
        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50, 1)

    def test_preserves_index(self, default_config):
        """DataFrame index should be preserved through predict."""
        rng = np.random.RandomState(42)
        idx = pd.Index([100, 200, 300, 400, 500])
        X = pd.DataFrame(rng.randn(5, 3), index=idx, columns=["a", "b", "c"])
        y = pd.DataFrame(
            {"target_hr": [10, 20, 30, 15, 25], "target_sb": [5, 3, 8, 2, 6]},
            index=idx,
        )
        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert list(preds.index) == list(idx)

    def test_fit_with_nullable_dtypes(self, default_config):
        """MTLForecaster handles DataFrames with nullable pandas dtypes."""
        X, y = _make_training_data(n_samples=50, n_features=5)
        X["feat_0"] = X["feat_0"].astype("Float64")
        X.loc[X.index[0], "feat_0"] = pd.NA

        model = MTLForecaster(default_config)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape[0] == len(X)


# ---------------------------------------------------------------------------
# TestEarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Tests for early stopping with eval_set."""

    @pytest.fixture()
    def train_val_data(self):
        X_train, y_train = _make_training_data(n_samples=100, seed=42)
        X_val, y_val = _make_training_data(n_samples=30, seed=99)
        return X_train, y_train, X_val, y_val

    def test_fit_with_eval_set(self, train_val_data):
        X_train, y_train, X_val, y_val = train_val_data
        config = _make_small_config(epochs=10)
        model = MTLForecaster(config)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        assert model.is_fitted_

    def test_early_stopping_fewer_epochs(self, train_val_data):
        """With short patience and many epochs, should stop before max."""
        X_train, y_train, X_val, y_val = train_val_data
        config = _make_small_config(epochs=200)
        config["model"]["early_stopping_patience"] = 3
        model = MTLForecaster(config)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        assert len(model.training_history_) < 200

    def test_fit_without_eval_set(self, train_val_data):
        """Training without eval_set should run all epochs."""
        X_train, y_train, _, _ = train_val_data
        config = _make_small_config(epochs=5)
        model = MTLForecaster(config)
        model.fit(X_train, y_train)  # No eval_set
        assert model.is_fitted_
        assert len(model.training_history_) == 5
        preds = model.predict(X_train)
        assert preds.shape[0] == 100


# ---------------------------------------------------------------------------
# TestModelPersistence
# ---------------------------------------------------------------------------


class TestModelPersistence:
    """Tests for save/load roundtrip."""

    @pytest.fixture()
    def fitted_model(self):
        X, y = _make_training_data(n_samples=60, n_features=5)
        config = _make_small_config()
        model = MTLForecaster(config)
        model.fit(X, y)
        return model, X

    def test_save_creates_file(self, fitted_model, tmp_path):
        model, _ = fitted_model
        path = model.save(tmp_path / "mtl")
        assert path.exists()
        assert path.name == "mtl_forecaster.pt"

    def test_save_unfitted_raises(self, tmp_path):
        model = MTLForecaster()
        with pytest.raises(RuntimeError, match="unfitted"):
            model.save(tmp_path / "mtl")

    def test_load_roundtrip(self, fitted_model, tmp_path):
        model, X = fitted_model
        model.save(tmp_path / "mtl")

        loaded = MTLForecaster.load(tmp_path / "mtl")
        preds_original = model.predict(X)
        preds_loaded = loaded.predict(X)

        pd.testing.assert_frame_equal(preds_original, preds_loaded)

    def test_load_preserves_metadata(self, fitted_model, tmp_path):
        model, _ = fitted_model
        model.save(tmp_path / "mtl")
        loaded = MTLForecaster.load(tmp_path / "mtl")

        assert loaded.seed == model.seed
        assert loaded.target_names_ == model.target_names_
        assert loaded.feature_names_ == model.feature_names_
        assert loaded.hidden_dims == model.hidden_dims
        assert loaded.head_dim == model.head_dim
        assert loaded.dropouts == model.dropouts
        assert loaded.is_fitted_


# ---------------------------------------------------------------------------
# TestLearnedTaskWeights
# ---------------------------------------------------------------------------


class TestLearnedTaskWeights:
    """Tests for uncertainty weight extraction."""

    @pytest.fixture()
    def fitted_model(self):
        X, y = _make_training_data(n_samples=60, n_features=5)
        config = _make_small_config()
        model = MTLForecaster(config)
        model.fit(X, y)
        return model

    def test_weights_structure(self, fitted_model):
        weights = fitted_model.get_learned_task_weights()
        assert isinstance(weights, dict)
        assert len(weights) == 6
        assert set(weights.keys()) == set(TARGET_COLUMNS)

    def test_weights_are_positive(self, fitted_model):
        weights = fitted_model.get_learned_task_weights()
        for name, w in weights.items():
            assert w > 0, f"{name} has non-positive weight {w}"

    def test_weights_unfitted_raises(self):
        model = MTLForecaster()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.get_learned_task_weights()


# ---------------------------------------------------------------------------
# TestMTLEvalPipeline
# ---------------------------------------------------------------------------


class TestMTLEvalPipeline:
    """Integration tests with the evaluation pipeline."""

    @pytest.fixture()
    def eval_data(self):
        X, y = _make_training_data(n_samples=80, n_features=5)
        config = _make_small_config(epochs=10)
        model = MTLForecaster(config)
        model.fit(X, y)
        preds = model.predict(X)
        return y, preds

    def test_metrics_computation(self, eval_data):
        y, preds = eval_data
        metrics = compute_metrics(y, preds, TARGET_DISPLAY)
        assert "per_target" in metrics
        assert "aggregate" in metrics
        assert len(metrics["per_target"]) == 6

    def test_naive_baseline_comparison(self, eval_data):
        y, preds = eval_data
        model_metrics = compute_metrics(y, preds, TARGET_DISPLAY)
        # Use actual values as "naive" predictions for testing
        naive_metrics = compute_naive_baseline(y, y, TARGET_DISPLAY)
        comparison = compare_to_baseline(model_metrics, naive_metrics)
        assert "per_target" in comparison
        assert "aggregate" in comparison

    def test_report_generation(self, eval_data, tmp_path):
        y, preds = eval_data
        metrics = compute_metrics(y, preds, TARGET_DISPLAY)
        # print_report should not raise
        print_report(metrics, model_name="MTL", split_name="test")
        # save_report should create file
        report_path = tmp_path / "mtl_report.json"
        save_report(metrics, model_name="MTL", split_name="test", path=report_path)
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert data["model"] == "MTL"


class TestLegacyResidualAndGatedHeads:
    """Coverage for legacy architecture add-ons from PR18."""

    def test_residual_block_projection_selection(self):
        same = ResidualBlock(8, 8, use_residual=True)
        diff = ResidualBlock(8, 16, use_residual=True)
        assert isinstance(same.proj, nn.Identity)
        assert isinstance(diff.proj, nn.Linear)

    def test_residual_block_disabled_is_plain_transform(self):
        block = ResidualBlock(8, 12, dropout=0.0, use_residual=False)
        x = torch.randn(4, 8)
        out = block(x)
        expected = block.dropout(block.relu(block.linear(x)))
        torch.testing.assert_close(out, expected)

    def test_gated_task_head_shape(self):
        head = GatedTaskHead(16, head_dim=8)
        x = torch.randn(5, 16)
        out = head(x)
        assert out.shape == (5, 1)

    def test_legacy_network_residual_and_gated_heads_forward(self):
        net = MTLNetwork(
            n_features=10,
            n_targets=4,
            hidden_dims=[16, 8],
            head_dim=4,
            dropouts=[0.1, 0.1],
            use_residual=True,
            use_gated_heads=True,
        )
        out = net(torch.randn(6, 10))
        assert len(out) == 4
        assert all(t.shape == (6, 1) for t in out)

    def test_legacy_forecaster_save_load_with_residual_and_gated(self, tmp_path):
        X, y = _make_training_data(n_samples=60, n_features=6)
        cfg = _make_small_config()
        cfg["model"]["use_residual"] = True
        cfg["model"]["use_gated_heads"] = True
        model = MTLForecaster(cfg)
        model.fit(X, y)
        model.save(tmp_path / "mtl")

        loaded = MTLForecaster.load(tmp_path / "mtl")
        assert loaded.use_residual is True
        assert loaded.use_gated_heads is True
        pd.testing.assert_frame_equal(model.predict(X), loaded.predict(X))


class TestLegacyTrainingAddOns:
    """Coverage for mixup/cosine/SWA additions on the legacy forecaster."""

    def test_mixup_enabled_legacy_fit_predict(self):
        X, y = _make_training_data(n_samples=80, n_features=6)
        cfg = _make_small_config()
        cfg["model"]["mixup_alpha"] = 0.2
        model = MTLForecaster(cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80, 6)
        assert preds.isna().sum().sum() == 0

    def test_legacy_cosine_scheduler_without_eval_runs_full_epochs(self):
        X, y = _make_training_data(n_samples=80, n_features=6)
        cfg = _make_small_config(epochs=6)
        cfg["model"]["lr_scheduler"] = {
            "type": "cosine_warm",
            "T_0": 2,
            "T_mult": 1,
            "eta_min": 1e-6,
        }
        model = MTLForecaster(cfg)
        model.fit(X, y)
        assert model.is_fitted_
        assert len(model.training_history_) == 6

    def test_legacy_swa_enabled_fit_predict(self):
        X, y = _make_training_data(n_samples=80, n_features=6)
        cfg = _make_small_config()
        cfg["model"]["swa"] = {
            "enabled": True,
            "lr": 1e-3,
            "epochs": 2,
            "anneal_epochs": 1,
        }
        model = MTLForecaster(cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80, 6)
        assert preds.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# TestTargetWinsorization
# ---------------------------------------------------------------------------


class TestTargetWinsorization:
    """Tests for optional target winsorization in MTLForecaster."""

    def test_winsorize_disabled_by_default(self):
        """MTLForecaster() should have target_winsorize_pct == 0.0 by default."""
        model = MTLForecaster()
        assert model.target_winsorize_pct == 0.0

    def test_winsorize_fit_predict_shape(self):
        """Training with target_winsorize_pct=2 should produce correct output shape."""
        X, y = _make_training_data(n_samples=100, n_features=10)
        cfg = _make_small_config()
        cfg["loss"] = {"type": "huber", "delta": 2.0, "target_winsorize_pct": 2}
        model = MTLForecaster(cfg)
        assert model.target_winsorize_pct == 2
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (100, 6)
        assert preds.isna().sum().sum() == 0

    def test_winsorize_save_load_roundtrip(self, tmp_path):
        """Save/load preserves target_winsorize_pct and predictions match."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config()
        cfg["loss"] = {"type": "huber", "delta": 2.0, "target_winsorize_pct": 2}
        model = MTLForecaster(cfg)
        model.fit(X, y)
        model.save(tmp_path / "mtl")

        loaded = MTLForecaster.load(tmp_path / "mtl")
        assert loaded.target_winsorize_pct == 2

        preds_original = model.predict(X)
        preds_loaded = loaded.predict(X)
        pd.testing.assert_frame_equal(preds_original, preds_loaded)


# ---------------------------------------------------------------------------
# TestTwoStageArchitecture
# ---------------------------------------------------------------------------


class TestTwoStageArchitecture:
    """Tests for the two-stage rate-to-count head architecture."""

    def test_two_stage_network_forward_shape(self):
        """Two-stage MTLNetwork returns 6 tensors of shape (batch, 1)."""
        net = MTLNetwork(n_features=10, n_targets=6, two_stage=True)
        x = torch.randn(8, 10)
        out = net(x)
        assert isinstance(out, list)
        assert len(out) == 6
        for t in out:
            assert t.shape == (8, 1)

    def test_two_stage_count_head_input_dim(self):
        """Count heads (indices 2-5) have in_features = backbone_out + 2."""
        net = MTLNetwork(
            n_features=10,
            n_targets=6,
            hidden_dims=[256, 128, 64],
            head_dim=32,
            dropouts=[0.3, 0.2, 0.1],
            two_stage=True,
        )
        # Rate heads (0, 1): first Linear should have in_features == 64
        for i in (0, 1):
            head = net.heads[i]
            first_linear = head[0]  # nn.Sequential: Linear, ReLU, Linear
            assert first_linear.in_features == 64, (
                f"Rate head {i} in_features={first_linear.in_features}, expected 64"
            )
        # Count heads (2-5): first Linear should have in_features == 66
        for i in (2, 3, 4, 5):
            head = net.heads[i]
            first_linear = head[0]
            assert first_linear.in_features == 66, (
                f"Count head {i} in_features={first_linear.in_features}, expected 66"
            )

    def test_two_stage_gradient_isolation(self):
        """Count-only loss must not backpropagate into rate head parameters."""
        net = MTLNetwork(
            n_features=10,
            n_targets=6,
            hidden_dims=[16, 8],
            head_dim=4,
            dropouts=[0.1, 0.1],
            two_stage=True,
        )
        x = torch.randn(4, 10)
        out = net(x)

        # Compute loss on count heads only (indices 2-5)
        count_loss = sum(o.sum() for o in out[2:])
        count_loss.backward()

        # Rate head parameters should have zero gradient
        for i in (0, 1):
            for name, param in net.heads[i].named_parameters():
                if param.grad is not None:
                    assert torch.all(param.grad == 0), (
                        f"Rate head {i} param '{name}' has non-zero gradient"
                    )

    def test_two_stage_disabled_matches_default(self):
        """With two_stage=False, parameter count matches default construction."""
        kwargs = dict(
            n_features=10,
            n_targets=6,
            hidden_dims=[16, 8],
            head_dim=4,
            dropouts=[0.1, 0.1],
        )
        net_default = MTLNetwork(**kwargs, two_stage=False)
        net_control = MTLNetwork(**kwargs)
        n_default = sum(p.numel() for p in net_default.parameters())
        n_control = sum(p.numel() for p in net_control.parameters())
        assert n_default == n_control

    def test_two_stage_forecaster_fit_predict(self):
        """Full MTLForecaster fit/predict with two_stage=True."""
        X, y = _make_training_data(n_samples=80, n_features=10)
        cfg = _make_small_config(epochs=5)
        cfg["model"]["two_stage"] = True
        model = MTLForecaster(cfg)
        assert model.two_stage is True
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80, 6)
        assert preds.isna().sum().sum() == 0

    def test_two_stage_save_load_roundtrip(self, tmp_path):
        """Save/load preserves two_stage flag and predictions match."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config()
        cfg["model"]["two_stage"] = True
        model = MTLForecaster(cfg)
        model.fit(X, y)
        model.save(tmp_path / "mtl")

        loaded = MTLForecaster.load(tmp_path / "mtl")
        assert loaded.two_stage is True

        preds_original = model.predict(X)
        preds_loaded = loaded.predict(X)
        pd.testing.assert_frame_equal(preds_original, preds_loaded)


# ---------------------------------------------------------------------------
# TestMTLEnsembleForecaster
# ---------------------------------------------------------------------------


def _make_ensemble_config(n_seeds: int = 3, epochs: int = 3) -> dict:
    """Config for fast ensemble unit tests."""
    return {
        "model": {
            "hidden_dims": [16, 8],
            "head_dim": 4,
            "dropouts": [0.1, 0.1],
            "batch_size": 16,
            "epochs": epochs,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "early_stopping_patience": 3,
            "lr_scheduler": {
                "patience": 2,
                "factor": 0.5,
                "min_lr": 1e-6,
            },
        },
        "seed": 42,
        "ensemble": {
            "n_seeds": n_seeds,
            "base_seed": 42,
        },
    }


class TestMTLEnsembleForecaster:
    """Tests for the multi-seed ensemble wrapper."""

    @pytest.fixture()
    def training_data(self):
        return _make_training_data(n_samples=80, n_features=8)

    def test_ensemble_creates_n_models(self, training_data):
        """n_seeds=3 creates 3 fitted models."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=3)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)
        assert len(model.models_) == 3
        assert model.is_fitted_

    def test_ensemble_predict_shape(self, training_data):
        """Output shape matches single model shape."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=2)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80, 6)
        assert list(preds.columns) == TARGET_COLUMNS

    def test_ensemble_predict_averages(self, training_data):
        """Prediction equals mean of individual member predictions."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=3)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)

        ensemble_preds = model.predict(X)
        individual_preds = [m.predict(X) for m in model.models_]
        manual_avg = individual_preds[0].copy()
        for p in individual_preds[1:]:
            manual_avg += p
        manual_avg /= len(individual_preds)

        pd.testing.assert_frame_equal(ensemble_preds, manual_avg)

    def test_ensemble_different_seeds(self, training_data):
        """Each ensemble member has a unique seed."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=3)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)

        seeds = [m.seed for m in model.models_]
        assert len(set(seeds)) == 3
        assert seeds == [42, 43, 44]

    def test_ensemble_save_load_roundtrip(self, training_data, tmp_path):
        """Save/load preserves predictions."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=2)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)
        preds_original = model.predict(X)

        model.save(tmp_path / "ensemble")
        loaded = MTLEnsembleForecaster.load(tmp_path / "ensemble")
        preds_loaded = loaded.predict(X)

        pd.testing.assert_frame_equal(preds_original, preds_loaded)
        assert loaded.n_seeds == 2
        assert loaded.is_fitted_
        assert len(loaded.models_) == 2

    def test_ensemble_learned_task_weights(self, training_data):
        """Returns averaged weights across members."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=2)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)

        weights = model.get_learned_task_weights()
        assert isinstance(weights, dict)
        assert len(weights) == 6
        for name, w in weights.items():
            assert w > 0, f"{name} has non-positive weight {w}"

        # Check that it's actually the average of individual weights
        w0 = model.models_[0].get_learned_task_weights()
        w1 = model.models_[1].get_learned_task_weights()
        for key in weights:
            expected = (w0[key] + w1[key]) / 2
            assert abs(weights[key] - expected) < 1e-6

    def test_ensemble_single_seed(self, training_data):
        """n_seeds=1 produces same shape as single model."""
        X, y = training_data
        cfg = _make_ensemble_config(n_seeds=1)
        model = MTLEnsembleForecaster(cfg)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (80, 6)
        assert len(model.models_) == 1


# ---------------------------------------------------------------------------
# TestThreeGroupArchitecture
# ---------------------------------------------------------------------------


class TestThreeGroupArchitecture:
    """Tests for the three-group (rate / count / speed) head architecture."""

    def test_speed_head_backbone_only_dim(self):
        """SB head (index 5) has in_features=64, HR/R/RBI heads have 66."""
        net = MTLNetwork(
            n_features=10,
            n_targets=6,
            hidden_dims=[256, 128, 64],
            head_dim=32,
            dropouts=[0.3, 0.2, 0.1],
            two_stage=True,
            speed_head_indices=[5],
        )
        # Rate heads (0, 1): in_features == 64
        for i in (0, 1):
            assert net.heads[i][0].in_features == 64, (
                f"Rate head {i} in_features={net.heads[i][0].in_features}, expected 64"
            )
        # Count heads (2, 3, 4): in_features == 66 (backbone + 2 rate preds)
        for i in (2, 3, 4):
            assert net.heads[i][0].in_features == 66, (
                f"Count head {i} in_features={net.heads[i][0].in_features}, expected 66"
            )
        # Speed head (5): in_features == 64 (backbone only)
        assert net.heads[5][0].in_features == 64, (
            f"Speed head 5 in_features={net.heads[5][0].in_features}, expected 64"
        )

    def test_three_group_forward_shape(self):
        """All 6 outputs are (batch, 1) with speed_head_indices=[5]."""
        net = MTLNetwork(
            n_features=10,
            n_targets=6,
            hidden_dims=[16, 8],
            head_dim=4,
            dropouts=[0.1, 0.1],
            two_stage=True,
            speed_head_indices=[5],
        )
        x = torch.randn(8, 10)
        out = net(x)
        assert isinstance(out, list)
        assert len(out) == 6
        for t in out:
            assert t.shape == (8, 1)

    def test_speed_head_uses_backbone_only(self):
        """SB output matches direct head(backbone(x)), not head(count_input)."""
        net = MTLNetwork(
            n_features=10,
            n_targets=6,
            hidden_dims=[16, 8],
            head_dim=4,
            dropouts=[0.1, 0.1],
            two_stage=True,
            speed_head_indices=[5],
        )
        net.eval()
        x = torch.randn(4, 10)
        with torch.no_grad():
            normed = net.batch_norm(x)
            shared = net.backbone(normed)
            direct_sb = net.heads[5](shared)
            full_out = net(x)
        torch.testing.assert_close(full_out[5], direct_sb)

    def test_empty_speed_indices_matches_legacy(self):
        """speed_head_indices=[] produces identical param count to legacy two_stage."""
        kwargs = dict(
            n_features=10,
            n_targets=6,
            hidden_dims=[16, 8],
            head_dim=4,
            dropouts=[0.1, 0.1],
            two_stage=True,
        )
        net_legacy = MTLNetwork(**kwargs)
        net_empty = MTLNetwork(**kwargs, speed_head_indices=[])
        n_legacy = sum(p.numel() for p in net_legacy.parameters())
        n_empty = sum(p.numel() for p in net_empty.parameters())
        assert n_legacy == n_empty

    def test_three_group_save_load_roundtrip(self, tmp_path):
        """Predictions match after save/load with speed_head_indices."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config()
        cfg["model"]["two_stage"] = True
        cfg["model"]["speed_head_indices"] = [5]
        model = MTLForecaster(cfg)
        model.fit(X, y)
        model.save(tmp_path / "mtl")

        loaded = MTLForecaster.load(tmp_path / "mtl")
        assert loaded.speed_head_indices == [5]
        assert loaded.two_stage is True

        preds_original = model.predict(X)
        preds_loaded = loaded.predict(X)
        pd.testing.assert_frame_equal(preds_original, preds_loaded)

    def test_load_legacy_no_speed_indices(self, tmp_path):
        """Old checkpoint without speed_head_indices defaults to []."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config()
        cfg["model"]["two_stage"] = True
        model = MTLForecaster(cfg)
        model.fit(X, y)
        model.save(tmp_path / "mtl")

        # Simulate legacy checkpoint by removing speed_head_indices
        model_path = tmp_path / "mtl" / "mtl_forecaster.pt"
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        del state["speed_head_indices"]
        torch.save(state, model_path)

        loaded = MTLForecaster.load(tmp_path / "mtl")
        assert loaded.speed_head_indices == []
        # Should still produce valid predictions
        preds = loaded.predict(X)
        assert preds.shape == (60, 6)

    def test_speed_heads_receive_rates_dim(self):
        """With speed_heads_receive_rates=True, speed heads get in_dim+2."""
        net = MTLNetwork(
            n_features=10, n_targets=6, hidden_dims=[16, 8],
            dropouts=[0.1, 0.1], two_stage=True,
            speed_head_indices=[5], speed_heads_receive_rates=True,
        )
        for i, head in enumerate(net.heads):
            first_linear = head[0] if isinstance(head, nn.Sequential) else head.gate
            if i >= 2:
                assert first_linear.in_features == 10, (
                    f"Head {i} should have in_dim+2=10 with speed_heads_receive_rates"
                )
            else:
                assert first_linear.in_features == 8

    def test_speed_heads_receive_rates_forward(self):
        """With speed_heads_receive_rates=True, SB head gets count_input."""
        net = MTLNetwork(
            n_features=10, n_targets=6, hidden_dims=[16, 8],
            dropouts=[0.1, 0.1], two_stage=True,
            speed_head_indices=[5], speed_heads_receive_rates=True,
        )
        x = torch.randn(4, 10)
        outputs = net(x)
        assert len(outputs) == 6
        for out in outputs:
            assert out.shape == (4, 1)

    def test_speed_heads_receive_rates_gradient_isolation(self):
        """Rate predictions are still detached even when feeding speed heads."""
        net = MTLNetwork(
            n_features=10, n_targets=6, hidden_dims=[16, 8],
            dropouts=[0.1, 0.1], two_stage=True,
            speed_head_indices=[5], speed_heads_receive_rates=True,
        )
        x = torch.randn(4, 10)
        outputs = net(x)
        # Backprop from SB head (index 5) only
        loss = outputs[5].sum()
        loss.backward()
        # Rate head parameters should have zero gradients
        for param in net.heads[0].parameters():
            assert param.grad is None or torch.all(param.grad == 0), \
                "Rate head gradients should be zero when only SB loss backprops"

    def test_speed_heads_receive_rates_save_load(self, tmp_path):
        """Checkpoint roundtrip preserves speed_heads_receive_rates flag."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config()
        cfg["model"]["two_stage"] = True
        cfg["model"]["speed_head_indices"] = [5]
        cfg["model"]["speed_heads_receive_rates"] = True
        model = MTLForecaster(cfg)
        model.fit(X, y)
        model.save(tmp_path / "mtl")

        loaded = MTLForecaster.load(tmp_path / "mtl")
        assert loaded.speed_heads_receive_rates is True
        preds = loaded.predict(X)
        assert preds.shape == (60, 6)


# ---------------------------------------------------------------------------
# TestRecencyWeighting
# ---------------------------------------------------------------------------


class TestRecencyWeighting:
    """Tests for sample-level recency weighting."""

    def test_dataset_with_sample_weights(self):
        """BatterDataset returns 3-tuple with correct weight values."""
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randn(10, 6).astype(np.float32)
        w = np.linspace(0.5, 1.5, 10).astype(np.float32)
        ds = BatterDataset(X, y, sample_weights=w)
        xi, yi, wi = ds[3]
        assert isinstance(wi, torch.Tensor)
        assert wi.item() == pytest.approx(w[3], abs=1e-6)

    def test_dataset_without_weights_compat(self):
        """BatterDataset returns 3-tuple with default weight 1.0."""
        X = np.random.randn(10, 4).astype(np.float32)
        y = np.random.randn(10, 6).astype(np.float32)
        ds = BatterDataset(X, y)
        xi, yi, wi = ds[0]
        assert wi.item() == pytest.approx(1.0)

    def test_dataset_inference_mode_unchanged(self):
        """Inference mode (no targets) returns just features."""
        X = np.random.randn(10, 4).astype(np.float32)
        ds = BatterDataset(X)
        out = ds[0]
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4,)

    def test_huber_loss_uniform_weights_matches_no_weights(self):
        """Passing all-ones weights produces same loss as no weights."""
        torch.manual_seed(42)
        preds = [torch.randn(8, 1) for _ in range(6)]
        targets = torch.randn(8, 6)
        ones = torch.ones(8)

        loss_fn = HuberMultiTaskLoss(n_tasks=6, delta=2.0)
        loss_no_w, _ = loss_fn(preds, targets)
        loss_with_w, _ = loss_fn(preds, targets, sample_weights=ones)
        assert loss_no_w.item() == pytest.approx(loss_with_w.item(), abs=1e-5)

    def test_huber_loss_nonuniform_weights(self):
        """Non-uniform weights shift the loss value."""
        torch.manual_seed(42)
        preds = [torch.randn(8, 1) for _ in range(6)]
        targets = torch.randn(8, 6)
        weights = torch.linspace(0.5, 1.5, 8)
        weights = weights / weights.mean()  # normalize

        loss_fn = HuberMultiTaskLoss(n_tasks=6, delta=2.0)
        loss_uniform, _ = loss_fn(preds, targets, sample_weights=torch.ones(8))
        loss_weighted, _ = loss_fn(preds, targets, sample_weights=weights)
        assert loss_uniform.item() != pytest.approx(loss_weighted.item(), abs=1e-4)

    def test_mse_loss_uniform_weights_matches_no_weights(self):
        """MultiTaskLoss: all-ones weights match no weights."""
        torch.manual_seed(42)
        preds = [torch.randn(8, 1) for _ in range(6)]
        targets = torch.randn(8, 6)
        ones = torch.ones(8)

        loss_fn = MultiTaskLoss(n_tasks=6)
        loss_no_w, _ = loss_fn(preds, targets)
        loss_with_w, _ = loss_fn(preds, targets, sample_weights=ones)
        assert loss_no_w.item() == pytest.approx(loss_with_w.item(), abs=1e-5)

    def test_recency_weight_computation(self):
        """Verify exponential decay formula and normalization."""
        seasons = np.array([2016, 2017, 2018, 2019, 2020, 2021, 2022])
        lam = 0.15
        max_s = seasons.max()
        raw = np.exp(-lam * (max_s - seasons))
        weights = raw / raw.mean()

        # Most recent season should have highest weight
        assert weights[-1] > weights[0]
        # Mean should be 1.0
        assert weights.mean() == pytest.approx(1.0, abs=1e-10)
        # Ratio newest/oldest should be exp(0.15 * 6) ≈ 2.46
        assert weights[-1] / weights[0] == pytest.approx(np.exp(lam * 6), abs=0.01)
        # Lambda=0 should give all 1.0
        raw_zero = np.exp(-0.0 * (max_s - seasons))
        weights_zero = raw_zero / raw_zero.mean()
        np.testing.assert_allclose(weights_zero, 1.0)

    def test_fit_with_recency_weights(self):
        """Integration: training with season info and recency decay completes."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config(epochs=3)
        cfg["model"]["recency_decay_lambda"] = 0.15
        model = MTLForecaster(cfg)
        seasons = np.repeat([2018, 2019, 2020], 20)
        model.fit(X, y, season=seasons)
        preds = model.predict(X)
        assert preds.shape == (60, 6)

    def test_fit_without_season_compat(self):
        """Existing fit path without season still works."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config(epochs=3)
        cfg["model"]["recency_decay_lambda"] = 0.15
        model = MTLForecaster(cfg)
        # No season arg — should train without error (weights disabled)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (60, 6)

    def test_fit_lambda_zero_disables_weighting(self):
        """Lambda=0 means no sample weighting regardless of season."""
        X, y = _make_training_data(n_samples=60, n_features=5)
        cfg = _make_small_config(epochs=3)
        cfg["model"]["recency_decay_lambda"] = 0.0
        model = MTLForecaster(cfg)
        seasons = np.repeat([2018, 2019, 2020], 20)
        model.fit(X, y, season=seasons)
        preds = model.predict(X)
        assert preds.shape == (60, 6)

"""Tests for plot functions and prediction helpers.

Covers:
- Plot function smoke tests (generate Figure objects without errors)
- Prediction script logic (round/clip, model configs)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.eval.plots import (
    plot_calibration_scatter,
    plot_model_comparison_bars,
    plot_pit_histogram,
    plot_residual_distributions,
    plot_training_curves,
    save_figure,
)
from src.features.registry import TARGET_COLUMNS, TARGET_DISPLAY

# ---------------------------------------------------------------------------
# Helpers: synthetic data and reports
# ---------------------------------------------------------------------------

_N_TARGETS = len(TARGET_COLUMNS)


def _make_predictions(n_samples: int = 50, seed: int = 42):
    """Generate synthetic y_true and y_pred DataFrames."""
    rng = np.random.RandomState(seed)

    y_true_arr = np.column_stack([
        rng.uniform(0.280, 0.420, n_samples),    # OBP
        rng.uniform(0.350, 0.600, n_samples),     # SLG
        rng.randint(5, 45, n_samples).astype(float),  # HR
        rng.randint(40, 120, n_samples).astype(float), # R
        rng.randint(30, 120, n_samples).astype(float), # RBI
        rng.randint(0, 40, n_samples).astype(float),   # SB
    ])

    # Predictions = true + noise
    noise = rng.randn(n_samples, _N_TARGETS) * np.array([0.02, 0.04, 5, 10, 10, 5])
    y_pred_arr = y_true_arr + noise

    y_true = pd.DataFrame(y_true_arr, columns=TARGET_COLUMNS)
    y_pred = pd.DataFrame(y_pred_arr, columns=TARGET_COLUMNS)
    return y_true, y_pred


def _make_report(model_name: str, seed: int = 42) -> dict:
    """Build a synthetic report dict matching the real JSON structure."""
    rng = np.random.RandomState(seed)
    per_target = {}
    rmse_vals, mae_vals, r2_vals, mape_vals = [], [], [], []

    for t in TARGET_DISPLAY:
        r = rng.uniform(0.02, 15.0)
        m = rng.uniform(0.01, 10.0)
        r2 = rng.uniform(0.1, 0.6)
        mp = rng.uniform(5.0, 50.0)
        per_target[t] = {"rmse": r, "mae": m, "r2": r2, "mape": mp}
        rmse_vals.append(r)
        mae_vals.append(m)
        r2_vals.append(r2)
        mape_vals.append(mp)

    aggregate = {
        "rmse": float(np.mean(rmse_vals)),
        "mae": float(np.mean(mae_vals)),
        "r2": float(np.mean(r2_vals)),
        "mape": float(np.mean(mape_vals)),
    }

    baseline_per = {}
    for t in TARGET_DISPLAY:
        baseline_per[t] = {
            "rmse": per_target[t]["rmse"] + rng.uniform(0.5, 2.0),
            "mae": per_target[t]["mae"] + rng.uniform(0.3, 1.5),
            "r2": per_target[t]["r2"] - rng.uniform(0.05, 0.2),
            "mape": per_target[t]["mape"] + rng.uniform(1.0, 5.0),
        }

    baseline_agg = {
        "rmse": float(np.mean([v["rmse"] for v in baseline_per.values()])),
        "mae": float(np.mean([v["mae"] for v in baseline_per.values()])),
        "r2": float(np.mean([v["r2"] for v in baseline_per.values()])),
        "mape": float(np.mean([v["mape"] for v in baseline_per.values()])),
    }

    # Build comparison
    comparison_per = {}
    for t in TARGET_DISPLAY:
        comparison_per[t] = {
            "rmse_improvement": baseline_per[t]["rmse"] - per_target[t]["rmse"],
            "mae_improvement": baseline_per[t]["mae"] - per_target[t]["mae"],
            "r2_improvement": per_target[t]["r2"] - baseline_per[t]["r2"],
            "mape_improvement": baseline_per[t]["mape"] - per_target[t]["mape"],
            "beats_baseline": True,
        }

    return {
        "model": model_name,
        "split": "test",
        "model_metrics": {"per_target": per_target, "aggregate": aggregate},
        "baseline_metrics": {"per_target": baseline_per, "aggregate": baseline_agg},
        "comparison": {
            "per_target": comparison_per,
            "aggregate": {
                "rmse_improvement": baseline_agg["rmse"] - aggregate["rmse"],
                "mae_improvement": baseline_agg["mae"] - aggregate["mae"],
                "r2_improvement": aggregate["r2"] - baseline_agg["r2"],
                "mape_improvement": baseline_agg["mape"] - aggregate["mape"],
                "targets_beaten": 6,
                "total_targets": 6,
            },
        },
    }


# ---------------------------------------------------------------------------
# Test class: Plot functions
# ---------------------------------------------------------------------------

class TestPlotFunctions:
    """Smoke tests for src.eval.plots functions."""

    def test_calibration_scatter_returns_figure(self):
        y_true, y_pred = _make_predictions()
        fig = plot_calibration_scatter(y_true, y_pred, TARGET_DISPLAY)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_calibration_scatter_has_correct_axes(self):
        y_true, y_pred = _make_predictions()
        fig = plot_calibration_scatter(y_true, y_pred, TARGET_DISPLAY)
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 6  # 2x3 grid
        plt.close(fig)

    def test_residual_distributions_returns_figure(self):
        y_true, y_pred = _make_predictions()
        fig = plot_residual_distributions(y_true, y_pred, TARGET_DISPLAY)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_residual_distributions_has_correct_axes(self):
        y_true, y_pred = _make_predictions()
        fig = plot_residual_distributions(y_true, y_pred, TARGET_DISPLAY)
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 6
        plt.close(fig)

    def test_model_comparison_bars_returns_figure(self):
        reports = {
            "MTL": _make_report("MTL", 42),
        }
        fig = plot_model_comparison_bars(reports, "rmse")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_model_comparison_bars_all_metrics(self):
        reports = {
            "MTL": _make_report("MTL", 42),
        }
        for metric in ["rmse", "mae", "r2", "mape"]:
            fig = plot_model_comparison_bars(reports, metric)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_training_curves_returns_figure(self):
        history = [
            {"epoch": i, "train_loss": 1.0 / (i + 1), "val_rmse": 0.5 / (i + 1)}
            for i in range(20)
        ]
        fig = plot_training_curves(history, "MTL")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_training_curves_without_val(self):
        history = [{"epoch": i, "train_loss": 1.0 / (i + 1)} for i in range(10)]
        fig = plot_training_curves(history, "MTL")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_figure_creates_file(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = save_figure(fig, tmp_path / "test.png")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_save_figure_creates_parents(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = save_figure(fig, tmp_path / "sub" / "dir" / "test.png")
        assert out.exists()

    def test_plot_pit_histogram_creates_file(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 200
        n_targets = 6
        taus = [0.05, 0.25, 0.50, 0.75, 0.95]
        target_names = ["OBP", "SLG", "HR/PA", "R/PA", "RBI/PA", "SB/PA"]

        # Synthetic but roughly well-calibrated: each tau q_i is the sample
        # quantile across rows, broadcast to all rows.
        y_true = rng.uniform(0.2, 0.5, size=(n, n_targets))
        quant_vals = np.quantile(y_true, taus, axis=0).T  # (n_targets, n_taus)
        y_quant = np.broadcast_to(quant_vals[None, :, :], (n, n_targets, len(taus))).copy()

        out_path = tmp_path / "pit_test.png"
        returned = plot_pit_histogram(y_true, y_quant, taus, target_names, out_path)
        assert returned == out_path
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_plot_pit_histogram_shape_validation(self, tmp_path):
        # Mismatched tau count vs quantile axis should raise.
        y_true = np.zeros((5, 2))
        y_quant = np.zeros((5, 2, 3))
        with pytest.raises(ValueError, match="3 quantiles but 2 taus"):
            plot_pit_histogram(
                y_true, y_quant, [0.25, 0.75], ["A", "B"], tmp_path / "bad.png",
            )


# ---------------------------------------------------------------------------
# Test class: Prediction helpers
# ---------------------------------------------------------------------------

class TestPredictionHelpers:
    """Tests for prediction logic in generate_projections."""

    def test_build_single_model_results(self):
        from scripts.generate_projections import _build_single_model_results

        predictions = pd.DataFrame({
            "target_obp": [0.350, 0.300],
            "target_slg": [0.500, 0.400],
            "target_hr": [30.0, 20.0],
            "target_r": [90.0, 70.0],
            "target_rbi": [85.0, 65.0],
            "target_sb": [15.0, 5.0],
        })
        result = _build_single_model_results(predictions, "MTL")
        assert "MTL OBP" in result
        assert "MTL SB" in result
        assert len(result) == 6
        np.testing.assert_array_equal(result["MTL HR"], [30.0, 20.0])

    def test_round_and_clip_rate_stats(self):
        from scripts.generate_projections import _round_and_clip

        results = pd.DataFrame({
            "MTL OBP": [0.34567, 0.29123],
            "MTL SLG": [0.51234, 0.39876],
            "MTL HR": [29.7, -2.3],
            "MTL R": [89.3, 70.8],
            "MTL RBI": [84.6, -1.0],
            "MTL SB": [14.7, 5.2],
        })
        _round_and_clip(results, "MTL")
        assert results["MTL OBP"].iloc[0] == 0.346
        assert results["MTL SLG"].iloc[1] == 0.399
        assert results["MTL HR"].iloc[1] == 0  # clipped from -2.3
        assert results["MTL RBI"].iloc[1] == 0  # clipped from -1.0

    def test_model_configs_complete(self):
        """MODEL_CONFIGS has the MTL model entry."""
        from src.models.utils import get_model_configs

        configs = get_model_configs()
        assert "mtl" in configs
        for key, info in configs.items():
            assert "class" in info
            assert "config_path" in info
            assert "model_dir" in info
            assert "display_name" in info

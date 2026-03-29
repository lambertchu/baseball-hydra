"""Tests for rolling-origin backtest utilities and metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.backtest import DEFAULT_BACKTEST_FOLDS, iter_backtest_splits, load_backtest_folds
from src.eval.metrics import paired_bootstrap_rmse_delta
from src.models.mtl import train as mtl_train


class TestBacktestFolds:
    def test_default_fold_count(self):
        folds = load_backtest_folds({})
        assert len(folds) == 5
        assert [f.name for f in folds] == ["A", "B", "C", "D", "E"]

    def test_default_fold_schedule(self):
        assert DEFAULT_BACKTEST_FOLDS[0].train_end == 2019
        assert DEFAULT_BACKTEST_FOLDS[-1].test_year == 2025

    def test_iter_backtest_splits_respects_temporal_order(self):
        df = pd.DataFrame(
            {
                "season": np.repeat(np.arange(2018, 2026), 3),
                "target_obp": np.random.rand(24),
                "target_slg": np.random.rand(24),
                "target_hr": np.random.rand(24),
                "target_r": np.random.rand(24),
                "target_rbi": np.random.rand(24),
                "target_sb": np.random.rand(24),
            }
        )
        folds = load_backtest_folds({})
        for fold, train_df, val_df, test_df in iter_backtest_splits(
            df,
            folds,
            target_cols=["target_obp", "target_slg", "target_hr", "target_r", "target_rbi", "target_sb"],
        ):
            assert train_df["season"].max() <= fold.train_end
            assert val_df["season"].nunique() <= 1
            if len(val_df) > 0:
                assert val_df["season"].iloc[0] == fold.val_year
            assert test_df["season"].nunique() <= 1
            if len(test_df) > 0:
                assert test_df["season"].iloc[0] == fold.test_year


class TestBootstrapDelta:
    def test_paired_bootstrap_rmse_delta_shape(self):
        y_true = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 3.0, 4.0, 5.0]})
        y_model = pd.DataFrame({"a": [1.1, 1.9, 3.1, 4.0], "b": [2.1, 2.9, 4.0, 5.1]})
        y_base = pd.DataFrame({"a": [0.8, 2.5, 2.5, 4.8], "b": [1.2, 3.8, 3.4, 6.0]})

        out = paired_bootstrap_rmse_delta(
            y_true=y_true,
            y_pred_model=y_model,
            y_pred_baseline=y_base,
            n_bootstrap=100,
            seed=42,
        )

        assert "delta_mean" in out
        assert "delta_ci95_low" in out
        assert "delta_ci95_high" in out
        assert out["n_bootstrap"] == 100


class TestBacktestEmptyFoldHandling:
    def test_evaluate_split_empty_raises_clear_error(self):
        with pytest.raises(ValueError, match="empty split DataFrame"):
            mtl_train._evaluate_split(  # type: ignore[arg-type]
                model=None,
                split_df=pd.DataFrame(),
                data_config={},
            )

    def test_run_backtest_skips_empty_fold_and_keeps_valid_fold(self, monkeypatch, tmp_path):
        # Minimal no-op data load/feature build path.
        base_df = pd.DataFrame({"season": [2020], "target_obp": [0.3]})
        monkeypatch.setattr(mtl_train.pd, "read_parquet", lambda _: base_df)
        monkeypatch.setattr(mtl_train, "build_features", lambda df, _: df)

        fold_a = mtl_train.load_backtest_folds({"backtest": {"folds": [
            {"name": "A", "train_end": 2019, "val_year": 2020, "test_year": 2021},
            {"name": "B", "train_end": 2020, "val_year": 2021, "test_year": 2022},
        ]}})[0]
        fold_b = mtl_train.load_backtest_folds({"backtest": {"folds": [
            {"name": "A", "train_end": 2019, "val_year": 2020, "test_year": 2021},
            {"name": "B", "train_end": 2020, "val_year": 2021, "test_year": 2022},
        ]}})[1]
        monkeypatch.setattr(mtl_train, "load_backtest_folds", lambda _: [fold_a, fold_b])

        # Fold A has empty test split; Fold B is valid.
        train = pd.DataFrame({"x": [1.0], "target_obp": [0.3]})
        val = pd.DataFrame({"x": [2.0], "target_obp": [0.31]})
        test_empty = pd.DataFrame({"x": [], "target_obp": []})
        test_ok = pd.DataFrame({"x": [3.0], "target_obp": [0.32]})

        def _iter(_df, _folds, _targets):
            yield fold_a, train, val, test_empty
            yield fold_b, train, val, test_ok

        monkeypatch.setattr(mtl_train, "iter_backtest_splits", _iter)
        monkeypatch.setattr(
            mtl_train,
            "extract_xy",
            lambda df, _: (df[["x"]], pd.DataFrame({"target_obp": df["target_obp"]})),
        )
        monkeypatch.setattr(mtl_train, "_fit_model", lambda **_: None)

        fake_metrics = {
            "per_target": {"OBP": {"rmse": 0.1, "mae": 0.1, "r2": 0.1, "mape": 0.1}},
            "aggregate": {"rmse": 0.1, "mae": 0.1, "r2": 0.1, "mape": 0.1},
        }
        monkeypatch.setattr(
            mtl_train,
            "_evaluate_split",
            lambda model, split_df, data_config: (
                fake_metrics,
                None,
                pd.DataFrame({"target_obp": [0.32]}),
                pd.DataFrame({"target_obp": [0.31]}),
                None,
            ),
        )
        monkeypatch.setattr(
            mtl_train,
            "build_report",
            lambda model_metrics, baseline_metrics, model_name, split_name: {
                "model_metrics": model_metrics,
                "baseline_metrics": baseline_metrics,
                "model_name": model_name,
                "split_name": split_name,
            },
        )
        monkeypatch.setattr(
            mtl_train,
            "summarize_backtest_metrics",
            lambda reports: {
                "aggregate": {"rmse_mean": 0.1},
                "per_target": {"OBP": {"rmse_mean": 0.1}},
            },
        )
        monkeypatch.setattr(mtl_train, "print_report", lambda *_, **__: None)
        monkeypatch.setattr(mtl_train, "_make_metadata", lambda **_: {})
        monkeypatch.setattr(mtl_train, "save_backtest_report", lambda **_: None)

        out = mtl_train.run_backtest(
            model_config={"output": {"backtest_report": str(tmp_path / "bt.json")}},
            data_config={},
            merged_data_path="ignored.parquet",
            model_config_path="configs/mtl.yaml",
            data_config_path="configs/data.yaml",
            device="cpu",
        )

        assert len(out["fold_reports"]) == 1
        assert out["fold_reports"][0]["fold"]["name"] == "B"
        assert out["metadata"]["evaluated_folds"] == ["B"]
        assert len(out["metadata"]["skipped_folds"]) == 1
        assert out["metadata"]["skipped_folds"][0]["name"] == "A"

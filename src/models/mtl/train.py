"""CLI training script for MTL models.

Supports:
- Legacy holdout training (2016-2022 train, 2023 val, 2024 test for 2025 target)
- Rolling-origin backtest mode
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.backtest import iter_backtest_splits, load_backtest_folds
from src.data.splits import SplitConfig, split_data
from src.eval.metrics import (
    compute_metrics,
    compute_naive_baseline,
    paired_bootstrap_rmse_delta,
    summarize_backtest_metrics,
)
from src.eval.report import build_report, print_report, save_backtest_report, save_report
from src.features.pipeline import build_features, extract_xy
from src.features.registry import TARGET_COLUMNS, TARGET_DISPLAY, TARGET_STATS
from src.models.mtl.model import MTLEnsembleForecaster, MTLForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PREV_YEAR_COLS = [f"prev_year_{s}" for s in TARGET_STATS]


def _load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _get_naive_predictions(df: pd.DataFrame) -> pd.DataFrame | None:
    available = [c for c in _PREV_YEAR_COLS if c in df.columns]
    if len(available) != len(_PREV_YEAR_COLS):
        return None
    return df[available].copy()


def _git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return None


def _file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _create_mtl_model(
    model_config: dict,
) -> MTLForecaster | MTLEnsembleForecaster:
    """Instantiate the appropriate MTL model from config."""
    ensemble_cfg = model_config.get("ensemble", {})
    n_seeds = ensemble_cfg.get("n_seeds", 0)

    if n_seeds > 1:
        logger.info("Training MTL ensemble (%d seeds) ...", n_seeds)
        return MTLEnsembleForecaster(model_config)
    else:
        logger.info("Training MTL ...")
        return MTLForecaster(model_config)


def _fit_model(
    model: MTLForecaster,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    train_df: pd.DataFrame,
) -> None:
    season = train_df["season"].values if "season" in train_df.columns else None
    model.fit(X_train, y_train, eval_set=(X_val, y_val), season=season)


def _evaluate_split(
    model: MTLForecaster,
    split_df: pd.DataFrame,
    data_config: dict,
) -> tuple[dict, dict | None, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    if split_df.empty:
        raise ValueError("Cannot evaluate an empty split DataFrame.")

    X_split, y_split = extract_xy(split_df, data_config)
    preds = model.predict(X_split)
    metrics = compute_metrics(y_split, preds, TARGET_DISPLAY)

    naive = _get_naive_predictions(split_df)
    baseline = None
    if naive is not None:
        # When rate_targets is enabled, y_split contains per-PA rates but naive
        # contains raw counts (prev_year_hr etc.). Convert naive to per-PA rates
        # using the player's prior-year PA so the baseline comparison is valid.
        rate_targets = data_config.get("rate_targets", False)
        if rate_targets and "pa" in split_df.columns:
            _count_stats = {"hr", "r", "rbi", "sb"}
            prior_pa = split_df["pa"].astype(float)
            for col in naive.columns:
                stat = col.replace("prev_year_", "")
                if stat in _count_stats:
                    naive[col] = np.where(
                        prior_pa.values > 0,
                        naive[col].values / prior_pa.values,
                        np.nan,
                    )
        valid = naive.notna().all(axis=1)
        if valid.sum() > 0:
            baseline = compute_naive_baseline(y_split[valid], naive[valid], TARGET_DISPLAY)
            metrics = compute_metrics(y_split[valid], preds[valid], TARGET_DISPLAY)
            y_split = y_split[valid]
            preds = preds[valid]

    return metrics, baseline, y_split, preds, naive


def _make_metadata(
    model_config_path: str | Path,
    data_config_path: str | Path,
    merged_data_path: str | Path,
    model_config: dict,
    data_config: dict,
) -> dict:
    return {
        "model_config_path": str(model_config_path),
        "data_config_path": str(data_config_path),
        "merged_data_path": str(merged_data_path),
        "model_config": model_config,
        "data_config": data_config,
        "git_commit": _git_commit_hash(),
        "data_sha256": _file_sha256(merged_data_path),
    }


def run_holdout(
    model_config: dict,
    data_config: dict,
    merged_data_path: str | Path,
    model_config_path: str | Path,
    data_config_path: str | Path,
    device: str | None = None,
) -> MTLForecaster | MTLEnsembleForecaster:
    logger.info("Loading merged data from %s", merged_data_path)
    df = pd.read_parquet(merged_data_path)

    logger.info("Building features ...")
    df = build_features(df, data_config)

    split_cfg = SplitConfig.from_dict(data_config["splits"])
    train_df, val_df, test_df = split_data(df, split_cfg, TARGET_COLUMNS)
    logger.info("Splits — train: %d  val: %d  test: %d", len(train_df), len(val_df), len(test_df))

    X_train, y_train = extract_xy(train_df, data_config)
    X_val, y_val = extract_xy(val_df, data_config)

    model = _create_mtl_model(model_config)

    if isinstance(model, MTLEnsembleForecaster):
        season = train_df["season"].values if "season" in train_df.columns else None
        model.fit(X_train, y_train, eval_set=(X_val, y_val), season=season)
    else:
        _fit_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            train_df=train_df,
        )

    task_weights = model.get_learned_task_weights()
    logger.info("Learned task weights:")
    for name, weight in task_weights.items():
        logger.info("  %s: %.4f", name, weight)

    logger.info("Evaluating on validation set ...")
    val_metrics, val_baseline, _, _, _ = _evaluate_split(model, val_df, data_config)
    print_report(val_metrics, val_baseline, model_name="MTL", split_name="validation")

    logger.info("Evaluating on test set ...")
    test_metrics, test_baseline, _, _, _ = _evaluate_split(model, test_df, data_config)
    print_report(test_metrics, test_baseline, model_name="MTL", split_name="test")

    model_dir = model_config.get("output", {}).get("model_dir", "data/models/mtl/")
    model.save(model_dir)

    report_path = model_config.get("output", {}).get("report", "data/reports/mtl_report.json")
    save_report(test_metrics, test_baseline, model_name="MTL", split_name="test", path=report_path)

    return model


def run_backtest(
    model_config: dict,
    data_config: dict,
    merged_data_path: str | Path,
    model_config_path: str | Path,
    data_config_path: str | Path,
    device: str | None = None,
) -> dict:
    logger.info("Loading merged data from %s", merged_data_path)
    df = pd.read_parquet(merged_data_path)

    logger.info("Building features ...")
    df = build_features(df, data_config)

    folds = load_backtest_folds(model_config)
    logger.info("Running backtest folds: %s", [f.name for f in folds])

    fold_reports: list[dict] = []
    y_all: list[pd.DataFrame] = []
    pred_all: list[pd.DataFrame] = []
    naive_all: list[pd.DataFrame] = []
    skipped_folds: list[dict[str, int | str]] = []

    for fold, train_df, val_df, test_df in iter_backtest_splits(df, folds, TARGET_COLUMNS):
        logger.info(
            "Fold %s - train<=%d val=%d test=%d | rows: train=%d val=%d test=%d",
            fold.name,
            fold.train_end,
            fold.val_year,
            fold.test_year,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
            logger.warning(
                "Skipping fold %s due to empty split(s): train=%d val=%d test=%d",
                fold.name,
                len(train_df),
                len(val_df),
                len(test_df),
            )
            skipped_folds.append(
                {
                    "name": fold.name,
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                }
            )
            continue

        X_train, y_train = extract_xy(train_df, data_config)
        X_val, y_val = extract_xy(val_df, data_config)

        model = _create_mtl_model(copy.deepcopy(model_config))

        if isinstance(model, MTLEnsembleForecaster):
            season = train_df["season"].values if "season" in train_df.columns else None
            model.fit(X_train, y_train, eval_set=(X_val, y_val), season=season)
        else:
            _fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                train_df=train_df,
        )

        metrics, baseline, y_true, preds, naive = _evaluate_split(model, test_df, data_config)
        report = build_report(
            model_metrics=metrics,
            baseline_metrics=baseline,
            model_name="MTL",
            split_name=f"fold_{fold.name}",
        )
        report["fold"] = {
            "name": fold.name,
            "train_end": fold.train_end,
            "val_year": fold.val_year,
            "test_year": fold.test_year,
        }
        fold_reports.append(report)

        y_all.append(y_true)
        pred_all.append(preds)
        if naive is not None:
            naive_all.append(naive.loc[y_true.index])

        print_report(metrics, baseline, model_name="MTL", split_name=f"fold_{fold.name}")

    if not fold_reports:
        raise ValueError(
            "No valid backtest folds were evaluated. "
            "All folds had at least one empty split (train/val/test)."
        )

    summary = summarize_backtest_metrics(fold_reports)

    bootstrap_delta = None
    if naive_all:
        y_concat = pd.concat(y_all, axis=0).reset_index(drop=True)
        p_concat = pd.concat(pred_all, axis=0).reset_index(drop=True)
        n_concat = pd.concat(naive_all, axis=0).reset_index(drop=True)
        bootstrap_delta = paired_bootstrap_rmse_delta(
            y_true=y_concat,
            y_pred_model=p_concat,
            y_pred_baseline=n_concat,
            n_bootstrap=2000,
            seed=model_config.get("seed", 42),
        )

    metadata = _make_metadata(
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        merged_data_path=merged_data_path,
        model_config=model_config,
        data_config=data_config,
    )

    # Optional fail-safe comparison against a tuned XGBoost backtest report.
    xgb_backtest_path = Path("data/reports/xgboost_backtest_report.json")
    promotable = True
    if xgb_backtest_path.exists():
        try:
            with open(xgb_backtest_path) as f:
                xgb_report = json.load(f)
            xgb_rmse_mean = xgb_report["summary"]["aggregate"]["rmse_mean"]
            mtl_rmse_mean = summary["aggregate"]["rmse_mean"]
            promotable = mtl_rmse_mean < xgb_rmse_mean
            metadata["xgboost_backtest_rmse_mean"] = xgb_rmse_mean
            metadata["mtl_backtest_rmse_mean"] = mtl_rmse_mean
        except Exception:
            logger.exception("Failed to parse xgboost backtest report; leaving promotable=True")

    metadata["promotable"] = promotable
    metadata["evaluated_folds"] = [fr.get("fold", {}).get("name") for fr in fold_reports]
    metadata["skipped_folds"] = skipped_folds

    out_path = model_config.get("output", {}).get(
        "backtest_report",
        "data/reports/mtl_backtest_report.json",
    )
    save_backtest_report(
        fold_reports=fold_reports,
        summary=summary,
        model_name="MTL",
        metadata=metadata,
        bootstrap_delta=bootstrap_delta,
        path=out_path,
    )

    return {
        "fold_reports": fold_reports,
        "summary": summary,
        "bootstrap_delta": bootstrap_delta,
        "metadata": metadata,
    }


def train(
    model_config_path: str | Path = "configs/mtl.yaml",
    data_config_path: str | Path = "configs/data.yaml",
    merged_data_path: str | Path | None = None,
    backtest: bool = False,
    device: str | None = None,
):
    model_config = _load_config(model_config_path)
    data_config = _load_config(data_config_path)

    if merged_data_path is None:
        merged_data_path = data_config.get("output", {}).get(
            "merged_dataset", "data/merged_batter_data.parquet"
        )

    if backtest:
        return run_backtest(
            model_config=model_config,
            data_config=data_config,
            merged_data_path=merged_data_path,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            device=device,
        )

    return run_holdout(
        model_config=model_config,
        data_config=data_config,
        merged_data_path=merged_data_path,
        model_config_path=model_config_path,
        data_config_path=data_config_path,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MTL neural network forecaster",
    )
    parser.add_argument("--config", type=str, default="configs/mtl.yaml", help="Path to MTL config YAML")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Path to data config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to merged Parquet file")
    parser.add_argument("--backtest", action="store_true", help="Run rolling-origin backtests instead of single holdout")
    parser.add_argument("--device", type=str, default=None, help="Training device override (cpu/cuda/mps)")
    args = parser.parse_args()

    train(
        model_config_path=args.config,
        data_config_path=args.data_config,
        merged_data_path=args.data,
        backtest=args.backtest,
        device=args.device,
    )


if __name__ == "__main__":
    main()

"""Ablation study for MTL Phase 3-4 changes.

Tests each hypothesis independently to isolate its effect on benchmark RMSE.
Loads data once, then runs 6 config variants through the benchmark pipeline.
"""
from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

from src.data.splits import get_production_data
from src.eval.metrics import rmse
from src.features.pipeline import build_features, extract_xy
from src.features.registry import TARGET_COLUMNS, TARGET_DISPLAY
from src.models.mtl.model import MTLEnsembleForecaster, MTLForecaster
from src.models.utils import align_features

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
_EVAL_YEARS = [2022, 2023, 2024, 2025]
_MIN_PA = 200
_SEED = 42

_AGING_FEATURES = ["age_delta_speed", "age_delta_power", "age_delta_patience"]


def _load_configs() -> tuple[dict, dict]:
    with open("configs/mtl.yaml") as f:
        model_config = yaml.safe_load(f)
    with open("configs/data.yaml") as f:
        data_config = yaml.safe_load(f)
    return model_config, data_config


def _make_variant(
    model_config: dict,
    data_config: dict,
    *,
    two_stage: bool = False,
    winsorize_pct: float = 0.0,
    n_seeds: int = 1,
    exclude_aging: bool = True,
    recency_decay_lambda: float = 0.0,
) -> tuple[dict, dict]:
    mc = deepcopy(model_config)
    dc = deepcopy(data_config)

    mc["model"]["two_stage"] = two_stage
    mc["loss"]["target_winsorize_pct"] = winsorize_pct
    mc["ensemble"] = {"n_seeds": n_seeds, "base_seed": 42}
    mc["model"]["recency_decay_lambda"] = recency_decay_lambda

    if exclude_aging:
        excl = dc.get("exclude_features", [])
        for f in _AGING_FEATURES:
            if f not in excl:
                excl.append(f)
        dc["exclude_features"] = excl

    return mc, dc


def main() -> None:
    print("Loading data and building features ...")
    base_mc, base_dc = _load_configs()

    df = pd.read_parquet("data/merged_batter_data.parquet")
    # Build features with ALL features (including aging) - we'll exclude at extract time
    df_featured = build_features(df, base_dc)

    # Define ablation variants
    variants = {
        "all_off": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=0, n_seeds=1, exclude_aging=True),
        "+H14_aging": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=0, n_seeds=1, exclude_aging=False),
        "+H13_winsorize": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=2, n_seeds=1, exclude_aging=True),
        "+H12_two_stage": _make_variant(base_mc, base_dc,
            two_stage=True, winsorize_pct=0, n_seeds=1, exclude_aging=True),
        "+H10_ensemble": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=0, n_seeds=5, exclude_aging=True),
        "+H15_recency_0.10": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=0, n_seeds=1, exclude_aging=True,
            recency_decay_lambda=0.10),
        "+H15_recency_0.15": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=0, n_seeds=1, exclude_aging=True,
            recency_decay_lambda=0.15),
        "+H15_recency_0.25": _make_variant(base_mc, base_dc,
            two_stage=False, winsorize_pct=0, n_seeds=1, exclude_aging=True,
            recency_decay_lambda=0.25),
        "all_on": _make_variant(base_mc, base_dc,
            two_stage=True, winsorize_pct=2, n_seeds=5, exclude_aging=False),
        "all_on_+recency": _make_variant(base_mc, base_dc,
            two_stage=True, winsorize_pct=0, n_seeds=5, exclude_aging=True,
            recency_decay_lambda=0.15),
    }

    results: dict[str, dict[str, float]] = {}
    for vname, (mc, dc) in variants.items():
        print(f"\n{'='*60}")
        print(f"  Running variant: {vname}")
        print(f"  two_stage={mc['model']['two_stage']}  "
              f"winsorize={mc['loss']['target_winsorize_pct']}  "
              f"n_seeds={mc['ensemble']['n_seeds']}  "
              f"aging={'included' if 'age_delta_speed' not in dc.get('exclude_features', []) else 'excluded'}")
        print(f"{'='*60}")

        all_true = []
        all_pred = []

        for year in _EVAL_YEARS:
            retrain_df, predict_df = get_production_data(
                df_featured, end_year=year - 1, target_cols=TARGET_COLUMNS,
            )
            predict_df = predict_df.dropna(subset=TARGET_COLUMNS)
            if "target_pa" in predict_df.columns:
                predict_df = predict_df[predict_df["target_pa"] >= _MIN_PA].copy()

            X_predict, y_true = extract_xy(predict_df, dc)

            needs_eval = retrain_df["season"].nunique() > 1
            if needs_eval:
                max_season = retrain_df["season"].max()
                train_part = retrain_df[retrain_df["season"] != max_season]
                eval_part = retrain_df[retrain_df["season"] == max_season]
                X_train, y_train = extract_xy(train_part, dc)
                X_eval, y_eval = extract_xy(eval_part, dc)
                eval_set = (X_eval, y_eval)
            else:
                train_part = retrain_df
                X_train, y_train = extract_xy(retrain_df, dc)
                eval_set = None

            cfg = deepcopy(mc)
            cfg["seed"] = _SEED

            ensemble_cfg = cfg.get("ensemble", {})
            n_seeds = ensemble_cfg.get("n_seeds", 0)

            if n_seeds > 1:
                model = MTLEnsembleForecaster(cfg)
            else:
                model = MTLForecaster(cfg)

            season = train_part["season"].values if "season" in train_part.columns else None
            model.fit(X_train, y_train, eval_set=eval_set, season=season)

            X_aligned = align_features(X_predict, model, vname)
            preds = model.predict(X_aligned)

            all_true.append(y_true.values)
            all_pred.append(preds.values)

            per_target_rmse = {
                TARGET_DISPLAY[i]: rmse(y_true.values[:, i], preds.values[:, i])
                for i in range(6)
            }
            mean_rmse = float(np.mean(list(per_target_rmse.values())))
            print(f"  {year}: n={len(y_true)}, mean RMSE={mean_rmse:.4f}")

        y_true_pooled = np.concatenate(all_true, axis=0)
        y_pred_pooled = np.concatenate(all_pred, axis=0)

        result = {}
        for i, target in enumerate(TARGET_DISPLAY):
            result[target] = float(rmse(y_true_pooled[:, i], y_pred_pooled[:, i]))
        result["Mean"] = float(np.mean([result[t] for t in TARGET_DISPLAY]))
        result["n"] = len(y_true_pooled)
        results[vname] = result

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  MTL Ablation Results (pooled {_EVAL_YEARS[0]}-{_EVAL_YEARS[-1]}, "
          f"n={results['all_off']['n']:.0f})")
    print(f"{'='*90}")

    header = f"{'Variant':20s}"
    for t in TARGET_DISPLAY:
        w = 8 if t in ("OBP", "SLG") else 8
        header += f"  {t:>{w}s}"
    header += f"  {'Mean':>8s}  {'Delta':>8s}"
    print(header)
    print("-" * len(header))

    baseline_mean = results["all_off"]["Mean"]
    for vname, res in results.items():
        row = f"{vname:20s}"
        for t in TARGET_DISPLAY:
            val = res[t]
            if t in ("OBP", "SLG"):
                row += f"  {val:8.4f}"
            else:
                row += f"  {val:8.2f}"
        delta = (res["Mean"] - baseline_mean) / baseline_mean * 100
        delta_str = "---" if vname == "all_off" else f"{delta:+.2f}%"
        row += f"  {res['Mean']:8.4f}  {delta_str:>8s}"
        print(row)

    # Print per-target deltas from baseline
    print(f"\n  Per-target delta from baseline (all_off):")
    header2 = f"{'Variant':20s}"
    for t in TARGET_DISPLAY:
        header2 += f"  {t:>8s}"
    print(header2)
    print("-" * len(header2))

    for vname, res in results.items():
        if vname == "all_off":
            continue
        row = f"{vname:20s}"
        for t in TARGET_DISPLAY:
            delta = (res[t] - results["all_off"][t]) / results["all_off"][t] * 100
            row += f"  {delta:+7.2f}%"
        print(row)


if __name__ == "__main__":
    main()

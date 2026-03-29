"""Evaluation metrics for stat forecast models.

Computes per-target and aggregate RMSE, MAE, R², and MAPE.
Also provides:
- naive baseline comparison (Y+1 = Y persistence)
- normalized RMSE utilities (for mixed-scale multi-task early stopping)
- paired bootstrap confidence intervals for RMSE deltas
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error.

    Excludes samples where y_true == 0 to avoid division by zero.
    Returns 0.0 if no valid samples remain.
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
    target_names: list[str] | None = None,
) -> dict:
    """Compute per-target and aggregate evaluation metrics.

    Parameters
    ----------
    y_true:
        Ground truth values. Shape (n_samples, n_targets).
    y_pred:
        Predicted values. Shape (n_samples, n_targets).
    target_names:
        Human-readable names for each target column. If None, uses
        column names from DataFrame or generic "target_0", "target_1", etc.

    Returns
    -------
    dict
        Nested dict with structure::

            {
                "per_target": {
                    "OBP": {"rmse": ..., "mae": ..., "r2": ..., "mape": ...},
                    ...
                },
                "aggregate": {"rmse": ..., "mae": ..., "r2": ..., "mape": ...},
            }
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    n_targets = y_true_arr.shape[1]

    if target_names is None:
        if isinstance(y_true, pd.DataFrame):
            target_names = list(y_true.columns)
        else:
            target_names = [f"target_{i}" for i in range(n_targets)]

    per_target: dict[str, dict[str, float]] = {}
    rmse_vals = []
    mae_vals = []
    r2_vals = []
    mape_vals = []

    for i, name in enumerate(target_names):
        yt = y_true_arr[:, i]
        yp = y_pred_arr[:, i]
        r = rmse(yt, yp)
        m = mae(yt, yp)
        r2 = r_squared(yt, yp)
        mp = mape(yt, yp)
        per_target[name] = {"rmse": r, "mae": m, "r2": r2, "mape": mp}
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

    return {"per_target": per_target, "aggregate": aggregate}


def compute_naive_baseline(
    y_true: pd.DataFrame | np.ndarray,
    y_naive: pd.DataFrame | np.ndarray,
    target_names: list[str] | None = None,
) -> dict:
    """Compute metrics for the naive persistence baseline (Y+1 = Y).

    Parameters
    ----------
    y_true:
        Ground truth next-season values. Shape (n_samples, n_targets).
    y_naive:
        Current-season values used as naive predictions (i.e. prev_year_*
        features or the raw current stats). Shape (n_samples, n_targets).
    target_names:
        Human-readable target names.

    Returns
    -------
    dict
        Same structure as ``compute_metrics`` output.
    """
    return compute_metrics(y_true, y_naive, target_names)


def compare_to_baseline(
    model_metrics: dict,
    baseline_metrics: dict,
) -> dict:
    """Compare model metrics against a baseline, computing improvement.

    Parameters
    ----------
    model_metrics:
        Output of ``compute_metrics`` for the model.
    baseline_metrics:
        Output of ``compute_metrics`` for the baseline.

    Returns
    -------
    dict
        Per-target and aggregate improvements. For RMSE/MAE/MAPE, positive
        means the model is better (lower error). For R², positive means
        the model explains more variance (higher is better).
    """
    per_target: dict[str, dict[str, float]] = {}
    for name in model_metrics["per_target"]:
        model = model_metrics["per_target"][name]
        baseline = baseline_metrics["per_target"][name]
        per_target[name] = {
            "rmse_improvement": baseline["rmse"] - model["rmse"],
            "mae_improvement": baseline["mae"] - model["mae"],
            "r2_improvement": model["r2"] - baseline["r2"],
            "mape_improvement": baseline["mape"] - model["mape"],
            "beats_baseline": model["rmse"] < baseline["rmse"],
        }

    model_agg = model_metrics["aggregate"]
    baseline_agg = baseline_metrics["aggregate"]
    aggregate = {
        "rmse_improvement": baseline_agg["rmse"] - model_agg["rmse"],
        "mae_improvement": baseline_agg["mae"] - model_agg["mae"],
        "r2_improvement": model_agg["r2"] - baseline_agg["r2"],
        "mape_improvement": baseline_agg["mape"] - model_agg["mape"],
        "targets_beaten": sum(
            1 for v in per_target.values() if v["beats_baseline"]
        ),
        "total_targets": len(per_target),
    }

    return {"per_target": per_target, "aggregate": aggregate}


def normalized_rmse(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
    scales: np.ndarray | list[float] | None = None,
    target_weights: np.ndarray | list[float] | None = None,
) -> dict:
    """Compute per-target RMSE normalized by target scale.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predictions shaped (n_samples, n_targets).
    scales:
        Per-target normalization scales. If None, uses std(y_true) with
        epsilon floor.
    target_weights:
        Optional per-target positive weights for aggregate score.

    Returns
    -------
    dict
        {
            "per_target_rmse": [...],
            "per_target_nrmse": [...],
            "aggregate_nrmse": float,
            "scales": [...],
        }
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
        yp = yp.reshape(-1, 1)

    n_targets = yt.shape[1]
    rmse_vals = np.array([rmse(yt[:, i], yp[:, i]) for i in range(n_targets)])

    if scales is None:
        scales_arr = np.std(yt, axis=0, ddof=0)
    else:
        scales_arr = np.asarray(scales, dtype=np.float64)
    scales_arr = np.where(scales_arr > 1e-8, scales_arr, 1.0)

    nrmse_vals = rmse_vals / scales_arr

    if target_weights is None:
        weights = np.ones(n_targets, dtype=np.float64)
    else:
        weights = np.asarray(target_weights, dtype=np.float64)
        weights = np.where(weights > 0.0, weights, 0.0)
        if np.all(weights == 0.0):
            weights = np.ones(n_targets, dtype=np.float64)
    weights = weights / weights.sum()

    aggregate = float(np.sum(nrmse_vals * weights))
    return {
        "per_target_rmse": rmse_vals.tolist(),
        "per_target_nrmse": nrmse_vals.tolist(),
        "aggregate_nrmse": aggregate,
        "scales": scales_arr.tolist(),
    }


def paired_bootstrap_rmse_delta(
    y_true: pd.DataFrame | np.ndarray,
    y_pred_model: pd.DataFrame | np.ndarray,
    y_pred_baseline: pd.DataFrame | np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap CI for aggregate RMSE delta.

    Delta is defined as ``baseline_rmse - model_rmse`` so positive values
    indicate model improvement.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp_m = np.asarray(y_pred_model, dtype=np.float64)
    yp_b = np.asarray(y_pred_baseline, dtype=np.float64)

    if yt.ndim == 1:
        yt = yt.reshape(-1, 1)
        yp_m = yp_m.reshape(-1, 1)
        yp_b = yp_b.reshape(-1, 1)

    n = yt.shape[0]
    if n == 0:
        return {
            "n_bootstrap": n_bootstrap,
            "delta_mean": 0.0,
            "delta_ci95_low": 0.0,
            "delta_ci95_high": 0.0,
            "baseline_rmse": 0.0,
            "model_rmse": 0.0,
            "improvement": False,
        }

    base_metric = normalized_rmse(yt, yp_b)["aggregate_nrmse"]
    model_metric = normalized_rmse(yt, yp_m)["aggregate_nrmse"]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_s = yt[idx]
        yp_m_s = yp_m[idx]
        yp_b_s = yp_b[idx]
        m = normalized_rmse(yt_s, yp_m_s)["aggregate_nrmse"]
        b = normalized_rmse(yt_s, yp_b_s)["aggregate_nrmse"]
        deltas[i] = b - m

    ci_low = float(np.quantile(deltas, 0.025))
    ci_high = float(np.quantile(deltas, 0.975))
    delta_mean = float(np.mean(deltas))

    return {
        "n_bootstrap": int(n_bootstrap),
        "delta_mean": delta_mean,
        "delta_ci95_low": ci_low,
        "delta_ci95_high": ci_high,
        "baseline_rmse": float(base_metric),
        "model_rmse": float(model_metric),
        "improvement": ci_low > 0.0,
    }


def summarize_backtest_metrics(
    fold_reports: list[dict],
) -> dict:
    """Summarize fold-wise backtest reports into mean/std aggregates."""
    if not fold_reports:
        return {"fold_count": 0, "aggregate": {}, "per_target": {}}

    agg_rmse = [r["model_metrics"]["aggregate"]["rmse"] for r in fold_reports]
    agg_mae = [r["model_metrics"]["aggregate"]["mae"] for r in fold_reports]
    agg_r2 = [r["model_metrics"]["aggregate"]["r2"] for r in fold_reports]
    agg_mape = [r["model_metrics"]["aggregate"]["mape"] for r in fold_reports]

    first_targets = fold_reports[0]["model_metrics"]["per_target"].keys()
    per_target: dict[str, dict[str, float]] = {}
    for t in first_targets:
        rmse_vals = [r["model_metrics"]["per_target"][t]["rmse"] for r in fold_reports]
        mae_vals = [r["model_metrics"]["per_target"][t]["mae"] for r in fold_reports]
        r2_vals = [r["model_metrics"]["per_target"][t]["r2"] for r in fold_reports]
        mape_vals = [r["model_metrics"]["per_target"][t]["mape"] for r in fold_reports]
        per_target[t] = {
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals)),
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals)),
            "r2_mean": float(np.mean(r2_vals)),
            "r2_std": float(np.std(r2_vals)),
            "mape_mean": float(np.mean(mape_vals)),
            "mape_std": float(np.std(mape_vals)),
        }

    return {
        "fold_count": len(fold_reports),
        "aggregate": {
            "rmse_mean": float(np.mean(agg_rmse)),
            "rmse_std": float(np.std(agg_rmse)),
            "mae_mean": float(np.mean(agg_mae)),
            "mae_std": float(np.std(agg_mae)),
            "r2_mean": float(np.mean(agg_r2)),
            "r2_std": float(np.std(agg_r2)),
            "mape_mean": float(np.mean(agg_mape)),
            "mape_std": float(np.std(agg_mape)),
        },
        "per_target": per_target,
    }

"""Plotting utilities for model evaluation and comparison.

Provides calibration scatter plots, residual distributions, model comparison
bar charts, training curve visualizations, and save utilities.

All plot functions return matplotlib Figure objects for flexible rendering.

Usage
-----
    from src.eval.plots import plot_calibration_scatter, save_figure

    fig = plot_calibration_scatter(y_true, y_pred, target_names, "Ridge")
    save_figure(fig, "data/reports/comparison/calibration_ridge.png")
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.eval.metrics import r_squared

matplotlib.use("Agg")

# Consistent model colors across all comparison plots
_MODEL_COLORS = {
    "MTL": "#DD8452",
}
_BASELINE_COLOR = "#CCCCCC"


def plot_calibration_scatter(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
    target_names: list[str],
    model_name: str = "Model",
) -> plt.Figure:
    """Predicted vs actual scatter plot per target (2x3 subplot grid).

    Parameters
    ----------
    y_true : shape (n_samples, n_targets)
    y_pred : shape (n_samples, n_targets)
    target_names : Display names for each target (length 6).
    model_name : Title prefix.

    Returns
    -------
    matplotlib Figure with 2x3 subplots, one per target.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    n_targets = y_true_arr.shape[1]
    nrows = (n_targets + 2) // 3
    ncols = min(n_targets, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_targets == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    for i in range(n_targets):
        ax = axes_flat[i]
        actual = y_true_arr[:, i]
        pred = y_pred_arr[:, i]

        ax.scatter(actual, pred, alpha=0.4, s=15, color="#4C72B0", edgecolors="none")

        # y=x reference line
        lo = min(actual.min(), pred.min())
        hi = max(actual.max(), pred.max())
        margin = (hi - lo) * 0.05
        ref = np.array([lo - margin, hi + margin])
        ax.plot(ref, ref, "r--", linewidth=1, alpha=0.7)

        # R-squared annotation
        r2 = r_squared(actual, pred)
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )

        name = target_names[i] if i < len(target_names) else f"Target {i}"
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)

    # Hide unused axes
    for j in range(n_targets, len(list(axes_flat))):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{model_name} — Predicted vs Actual", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_residual_distributions(
    y_true: pd.DataFrame | np.ndarray,
    y_pred: pd.DataFrame | np.ndarray,
    target_names: list[str],
    model_name: str = "Model",
) -> plt.Figure:
    """Residual histograms per target (2x3 subplot grid).

    Each subplot: histogram of (actual - predicted) with KDE overlay,
    mean and std annotated. Vertical dashed line at 0.

    Parameters
    ----------
    y_true : shape (n_samples, n_targets)
    y_pred : shape (n_samples, n_targets)
    target_names : Display names for each target.
    model_name : Title prefix.

    Returns
    -------
    matplotlib Figure with 2x3 subplots.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    n_targets = y_true_arr.shape[1]
    nrows = (n_targets + 2) // 3
    ncols = min(n_targets, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_targets == 1:
        axes = np.array([axes])
    axes_flat = axes.flat

    for i in range(n_targets):
        ax = axes_flat[i]
        residuals = y_true_arr[:, i] - y_pred_arr[:, i]

        sns.histplot(residuals, kde=True, ax=ax, color="#4C72B0", alpha=0.6)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)

        mu = np.mean(residuals)
        sigma = np.std(residuals)
        ax.text(
            0.95,
            0.95,
            f"mean={mu:.3f}\nstd={sigma:.3f}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.5},
        )

        name = target_names[i] if i < len(target_names) else f"Target {i}"
        ax.set_xlabel(f"Residual ({name})")
        ax.set_title(name)

    for j in range(n_targets, len(list(axes_flat))):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"{model_name} — Residual Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_model_comparison_bars(
    reports: dict[str, dict],
    metric: str = "rmse",
    target_names: list[str] | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing models across targets for one metric.

    Parameters
    ----------
    reports : dict mapping model name -> report JSON dict. Each has structure
        ``{model_metrics: {per_target: {TARGET: {rmse, mae, r2, mape}},
        aggregate: {...}}}``.
    metric : one of ``"rmse"``, ``"mae"``, ``"r2"``, ``"mape"``.
    target_names : subset of targets to show (default: all from first report).

    Returns
    -------
    Figure with grouped bars: one group per target + Aggregate.
    """
    model_names = list(reports.keys())
    first_report = reports[model_names[0]]

    if target_names is None:
        target_names = list(first_report["model_metrics"]["per_target"].keys())

    labels = target_names + ["Aggregate"]
    n_groups = len(labels)
    n_models = len(model_names)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 6))
    x = np.arange(n_groups)
    width = 0.7 / n_models

    for j, model_name in enumerate(model_names):
        report = reports[model_name]
        values = []
        for t in target_names:
            values.append(report["model_metrics"]["per_target"][t][metric])
        values.append(report["model_metrics"]["aggregate"][metric])

        color = _MODEL_COLORS.get(model_name, f"C{j}")
        offset = (j - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, color=color, alpha=0.85)

    # Baseline bars (hatched)
    if "baseline_metrics" in first_report:
        baseline_values = []
        baseline = first_report["baseline_metrics"]
        for t in target_names:
            baseline_values.append(baseline["per_target"][t][metric])
        baseline_values.append(baseline["aggregate"][metric])

        offset = (n_models - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset,
            baseline_values,
            width,
            label="Naive Baseline",
            color=_BASELINE_COLOR,
            alpha=0.7,
            hatch="//",
            edgecolor="gray",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)

    metric_labels = {"rmse": "RMSE", "mae": "MAE", "r2": "R²", "mape": "MAPE (%)"}
    ylabel = metric_labels.get(metric, metric.upper())
    ax.set_ylabel(ylabel)
    ax.set_title(f"Model Comparison — {ylabel} by Target")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig


def plot_training_curves(
    history: list[dict],
    model_name: str = "MTL",
) -> plt.Figure:
    """Training loss and validation RMSE curves over epochs.

    Parameters
    ----------
    history : list of epoch dicts with keys ``'epoch'``, ``'train_loss'``,
        and optionally ``'val_rmse'``, ``'lr'``.
    model_name : Display name for the title.

    Returns
    -------
    Figure with dual y-axes: train loss (left) and val RMSE (right).
    """
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = "#4C72B0"
    ax1.plot(epochs, train_loss, color=color1, linewidth=1.5, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    has_val = any("val_rmse" in h for h in history)
    if has_val:
        val_rmse = [h.get("val_rmse", np.nan) for h in history]
        ax2 = ax1.twinx()
        color2 = "#DD8452"
        ax2.plot(
            epochs, val_rmse, color=color2, linewidth=1.5, label="Val RMSE", linestyle="--"
        )
        ax2.set_ylabel("Validation RMSE", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    ax1.set_title(f"{model_name} — Training Curves")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    path: str | Path,
    dpi: int = 150,
) -> Path:
    """Save a figure to disk, creating parent directories if needed.

    Parameters
    ----------
    fig : matplotlib Figure to save.
    path : Output file path (e.g. ``"plots/calibration.png"``).
    dpi : Resolution in dots per inch.

    Returns
    -------
    Path to the saved file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out

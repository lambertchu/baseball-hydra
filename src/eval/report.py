"""Generate evaluation reports (JSON + console table).

Formats model evaluation results for both programmatic use (JSON) and
human-readable console output.

Usage
-----
    from src.eval.report import print_report, save_report

    print_report(metrics, baseline_metrics, model_name="Ridge")
    save_report(metrics, baseline_metrics, path="data/reports/regression_report.json")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.eval.metrics import compare_to_baseline

logger = logging.getLogger(__name__)


def build_report(
    model_metrics: dict,
    baseline_metrics: dict | None = None,
    model_name: str = "Model",
    split_name: str = "test",
) -> dict:
    """Build a structured evaluation report.

    Parameters
    ----------
    model_metrics:
        Output of ``compute_metrics`` for the model.
    baseline_metrics:
        Output of ``compute_metrics`` for the naive baseline. If None,
        comparison section is omitted.
    model_name:
        Name of the model for display.
    split_name:
        Name of the evaluation split (e.g. "val", "test").

    Returns
    -------
    dict
        Structured report with model metrics, optional baseline comparison.
    """
    report: dict = {
        "model": model_name,
        "split": split_name,
        "model_metrics": model_metrics,
    }

    if baseline_metrics is not None:
        report["baseline_metrics"] = baseline_metrics
        report["comparison"] = compare_to_baseline(model_metrics, baseline_metrics)

    return report


def print_report(
    model_metrics: dict,
    baseline_metrics: dict | None = None,
    model_name: str = "Model",
    split_name: str = "test",
) -> None:
    """Print a formatted evaluation report to the console.

    Parameters
    ----------
    model_metrics:
        Output of ``compute_metrics`` for the model.
    baseline_metrics:
        Output of ``compute_metrics`` for the naive baseline.
    model_name:
        Name of the model for display.
    split_name:
        Name of the evaluation split.
    """
    header = f"{model_name} — {split_name} set evaluation"
    print(f"\n{'=' * 70}")
    print(f"  {header}")
    print(f"{'=' * 70}")

    # Per-target model metrics
    targets = list(model_metrics["per_target"].keys())
    print(f"\n  {'Target':<12} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}")
    print(f"  {'-' * 48}")
    for t in targets:
        m = model_metrics["per_target"][t]
        print(f"  {t:<12} {m['rmse']:8.4f} {m['mae']:8.4f} {m['r2']:8.4f} {m['mape']:7.1f}%")
    agg = model_metrics["aggregate"]
    print(f"  {'-' * 48}")
    print(f"  {'Mean':<12} {agg['rmse']:8.4f} {agg['mae']:8.4f} {agg['r2']:8.4f} {agg['mape']:7.1f}%")

    # Baseline comparison
    if baseline_metrics is not None:
        comparison = compare_to_baseline(model_metrics, baseline_metrics)
        print(f"\n  vs. Naive Baseline (Y+1 = Y)")
        print(f"  {'Target':<12} {'ΔRMSE':>8} {'ΔMAE':>8} {'ΔR²':>8} {'Beats?':>8}")
        print(f"  {'-' * 48}")
        for t in targets:
            c = comparison["per_target"][t]
            beat = "Yes" if c["beats_baseline"] else "No"
            print(
                f"  {t:<12} "
                f"{c['rmse_improvement']:+8.4f} "
                f"{c['mae_improvement']:+8.4f} "
                f"{c['r2_improvement']:+8.4f} "
                f"{beat:>8}"
            )
        ca = comparison["aggregate"]
        print(f"  {'-' * 48}")
        print(
            f"  Targets beaten: {ca['targets_beaten']}/{ca['total_targets']}  "
            f"Mean ΔRMSE: {ca['rmse_improvement']:+.4f}"
        )

    print(f"{'=' * 70}\n")


def save_report(
    model_metrics: dict,
    baseline_metrics: dict | None = None,
    model_name: str = "Model",
    split_name: str = "test",
    path: str | Path = "data/reports/report.json",
) -> Path:
    """Save the evaluation report as JSON.

    Parameters
    ----------
    model_metrics:
        Output of ``compute_metrics`` for the model.
    baseline_metrics:
        Output of ``compute_metrics`` for the naive baseline.
    model_name:
        Name of the model.
    split_name:
        Name of the evaluation split.
    path:
        Output path for the JSON report.

    Returns
    -------
    Path
        Path to the saved report file.
    """
    report = build_report(model_metrics, baseline_metrics, model_name, split_name)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved → %s", out)
    return out


def build_backtest_report(
    fold_reports: list[dict],
    summary: dict,
    model_name: str = "Model",
    metadata: dict | None = None,
    bootstrap_delta: dict | None = None,
) -> dict:
    """Build a rolling-origin backtest report."""
    report: dict = {
        "model": model_name,
        "evaluation": "rolling_origin_backtest",
        "fold_reports": fold_reports,
        "summary": summary,
    }
    if bootstrap_delta is not None:
        report["bootstrap_delta"] = bootstrap_delta
    if metadata is not None:
        report["metadata"] = metadata
    return report


def save_backtest_report(
    fold_reports: list[dict],
    summary: dict,
    path: str | Path = "data/reports/mtl_backtest_report.json",
    model_name: str = "Model",
    metadata: dict | None = None,
    bootstrap_delta: dict | None = None,
) -> Path:
    """Save rolling-origin backtest report as JSON."""
    report = build_backtest_report(
        fold_reports=fold_reports,
        summary=summary,
        model_name=model_name,
        metadata=metadata,
        bootstrap_delta=bootstrap_delta,
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Backtest report saved → %s", out)
    return out

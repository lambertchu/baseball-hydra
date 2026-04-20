"""Rolling ROS (rest-of-season) benchmark for the in-season projection pipeline.

Evaluates baselines at per-player PA checkpoints (50 / 100 / 200 / 400) across
2023-2025 weekly snapshots. Baselines (all emit rate predictions for the
six ROS targets: OBP, SLG, HR/PA, R/PA, RBI/PA, SB/PA):

* **persist_observed** — predict ROS rate = observed ytd rate. No model needed.
* **frozen_preseason** — predict ROS rate = preseason MTL rate. Requires a
  preseason predictions parquet per year (either cached or regenerated).
* **marcel_blend** — PA-weighted blend of ytd and preseason rates with a
  prior-weight ``prior_pa`` (default 200). Requires preseason predictions.

Usage
-----
    uv run python scripts/benchmark_ros.py --years 2023 2024 2025
    uv run python scripts/benchmark_ros.py --years 2024 --include persist_observed

    # Retrain preseason MTL for each year and cache (expensive):
    uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --retrain
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.splits import get_production_data
from src.eval.metrics import compute_metrics
from src.eval.ros_metrics import (
    DEFAULT_PA_CHECKPOINTS,
    ROS_RATE_TARGETS,
    ROS_TARGET_DISPLAY,
    ROS_TARGET_STATS,
    ROS_YTD_RATES,
    pa_checkpoint_rows,
)
from src.features.pipeline import build_features, extract_xy
from src.features.registry import TARGET_COLUMNS
from src.models.utils import align_features, get_model_configs, train_model_for_year

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = get_model_configs()

_BASELINES_NO_PRESEASON = ("persist_observed",)
_BASELINES_NEED_PRESEASON = ("frozen_preseason", "marcel_blend")
ALL_BASELINES = (*_BASELINES_NO_PRESEASON, *_BASELINES_NEED_PRESEASON)

_BASELINE_DISPLAY = {
    "persist_observed": "PersistObs",
    "frozen_preseason": "FrozenPre",
    "marcel_blend": "MarcelBlend",
}

_DEFAULT_CACHE_DIR = Path("data/reports/benchmark_ros/preseason")
_DEFAULT_OUTPUT_DIR = Path("data/reports/benchmark_ros")


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------


def load_weekly_snapshots(
    years: list[int],
    raw_dir: str | Path = "data/raw",
) -> pd.DataFrame:
    """Concatenate ``weekly_snapshots_{year}.parquet`` for each year."""
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for y in sorted(years):
        path = raw_dir / f"weekly_snapshots_{y}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing. Run: uv run python -m src.data.build_snapshots --seasons {y}"
            )
        df = pd.read_parquet(path)
        frames.append(df)
        logger.info("Loaded %s (%d rows)", path, len(df))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Preseason prediction cache
# ---------------------------------------------------------------------------


def _retrain_preseason_for_year(
    eval_year: int,
    df_featured: pd.DataFrame,
    data_config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Retrain MTL for year ``eval_year`` and return its preseason rate predictions.

    Delegates the train/eval-carve-out/ensemble logic to the shared
    ``train_model_for_year`` helper so any changes stay in sync with the
    preseason benchmark.
    """
    retrain_df, predict_df = get_production_data(
        df_featured, end_year=eval_year - 1, target_cols=TARGET_COLUMNS,
    )
    model = train_model_for_year("mtl", retrain_df, data_config, seed=seed)

    info = MODEL_CONFIGS["mtl"]
    X_predict, _ = extract_xy(predict_df, data_config)
    X_aligned = align_features(X_predict, model, info["display_name"])
    preds = model.predict(X_aligned)

    # mlbam_id is the weekly-snapshot join key; idfg kept for debugging only.
    # Align on index rather than positionally: ``MTLForecaster.predict``
    # preserves the input DataFrame's index, so this is row-correct even if a
    # future ``extract_xy`` variant drops rows during feature extraction.
    out = preds.copy()
    for id_col in ("mlbam_id", "idfg"):
        if id_col in predict_df.columns:
            out[id_col] = predict_df.loc[out.index, id_col].values
    out["season"] = eval_year
    return out


def load_or_generate_preseason_cache(
    year: int,
    cache_dir: Path,
    df_featured: pd.DataFrame | None,
    data_config: dict | None,
    retrain: bool,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Return the preseason MTL rate predictions for year ``year`` as a DataFrame.

    Uses a per-year parquet cache. Returns ``None`` if the cache is missing
    and ``retrain`` is False (callers should then skip preseason-dependent
    baselines).
    """
    cache_path = cache_dir / f"mtl_preseason_{year}.parquet"
    if cache_path.exists():
        logger.info("  Loaded preseason cache %s", cache_path)
        return pd.read_parquet(cache_path)

    if not retrain:
        logger.warning(
            "  Missing preseason cache for %d (%s); skipping preseason-dependent "
            "baselines. Pass --retrain to regenerate.",
            year, cache_path,
        )
        return None

    if df_featured is None or data_config is None:
        raise RuntimeError(
            "Cannot retrain without df_featured and data_config",
        )
    logger.info("  Retraining preseason MTL for %d …", year)
    preds = _retrain_preseason_for_year(year, df_featured, data_config, seed=seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache_path, engine="pyarrow", compression="zstd", index=False)
    logger.info("  Cached preseason predictions → %s", cache_path)
    return preds


# ---------------------------------------------------------------------------
# Baseline predictions
# ---------------------------------------------------------------------------


def _ytd_rate_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-target ytd rate DataFrame aligned with ``ROS_RATE_TARGETS``."""
    missing = [c for c in ROS_YTD_RATES if c not in rows.columns]
    if missing:
        raise KeyError(f"Snapshot rows missing ytd rate columns: {missing}")
    out = pd.DataFrame({tgt: rows[ytd].values for tgt, ytd in zip(ROS_RATE_TARGETS, ROS_YTD_RATES)})
    return out


def _preseason_rate_matrix(
    rows: pd.DataFrame,
    preseason: pd.DataFrame,
    id_col: str = "mlbam_id",
) -> pd.DataFrame | None:
    """Broadcast preseason rate predictions onto the checkpoint row order.

    Returns ``None`` when the join key is missing or all predictions are NA;
    returns a DataFrame with ``ROS_RATE_TARGETS`` columns (NaN for unmatched
    players) otherwise.
    """
    if id_col not in rows.columns or id_col not in preseason.columns:
        return None

    pre_target_cols = [f"target_{s}" for s in ROS_TARGET_STATS]
    missing_cols = [c for c in pre_target_cols if c not in preseason.columns]
    if missing_cols:
        logger.warning(
            "Preseason cache missing columns %s — cannot build preseason baseline",
            missing_cols,
        )
        return None

    pre_indexed = preseason.drop_duplicates(subset=[id_col]).set_index(id_col)[pre_target_cols]
    aligned = pre_indexed.reindex(rows[id_col].values).reset_index(drop=True)
    aligned.columns = list(ROS_RATE_TARGETS)
    return aligned


def predict_persist_observed(rows: pd.DataFrame) -> pd.DataFrame:
    """Persist-observed baseline: ``ros_rate = ytd_rate``."""
    return _ytd_rate_matrix(rows)


def predict_frozen_preseason(
    rows: pd.DataFrame,
    preseason: pd.DataFrame,
    id_col: str = "mlbam_id",
) -> pd.DataFrame | None:
    """Frozen-preseason baseline: ``ros_rate = preseason_rate`` (ignores ytd)."""
    return _preseason_rate_matrix(rows, preseason, id_col=id_col)


def predict_marcel_blend(
    rows: pd.DataFrame,
    preseason: pd.DataFrame,
    prior_pa: float = 200.0,
    id_col: str = "mlbam_id",
    preseason_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Marcel-style blend: ``(ytd_pa·ytd + prior_pa·pre) / (ytd_pa + prior_pa)``.

    Pass an already-aligned ``preseason_matrix`` (from a prior call to
    ``predict_frozen_preseason`` on the same rows) to skip rebuilding the
    index-and-reindex inside this function.
    """
    pre = preseason_matrix
    if pre is None:
        pre = _preseason_rate_matrix(rows, preseason, id_col=id_col)
    if pre is None:
        return None
    ytd = _ytd_rate_matrix(rows)
    ytd_pa = rows["pa_ytd"].astype(float).values.reshape(-1, 1)
    w_obs = ytd_pa / (ytd_pa + prior_pa)
    w_pre = prior_pa / (ytd_pa + prior_pa)
    blended = ytd.values * w_obs + pre.values * w_pre
    return pd.DataFrame(blended, columns=list(ROS_RATE_TARGETS))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _ros_target_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Ground-truth ROS rate DataFrame aligned with ``ROS_RATE_TARGETS``."""
    missing = [c for c in ROS_RATE_TARGETS if c not in rows.columns]
    if missing:
        raise KeyError(f"Snapshot rows missing ROS target columns: {missing}")
    return rows[list(ROS_RATE_TARGETS)].reset_index(drop=True)


def evaluate_checkpoint(
    checkpoint_rows: pd.DataFrame,
    baselines: list[str],
    preseason: pd.DataFrame | None,
    prior_pa: float,
    min_ros_pa: int,
) -> dict:
    """Compute point metrics for each baseline at one PA checkpoint.

    Filters to rows where (a) the ROS outcome is well-defined (non-NaN
    targets, ``ros_pa >= min_ros_pa``) and (b) every active baseline has a
    non-NaN prediction.

    Returns ``{"n_players": int, "y_true": np.ndarray, "systems": {baseline: {"y_pred": np.ndarray, "metrics": dict}}}``,
    or ``{"n_players": 0, "systems": {}}`` when no baselines run or no rows survive filtering.
    ``y_true`` is retained at the top level so ``pool_by_threshold`` can
    concatenate ground-truth once across years instead of per-system.
    """
    # Per-baseline predictions. Build the preseason→rows matrix at most once:
    # both frozen_preseason and marcel_blend reuse the same alignment.
    predictions: dict[str, pd.DataFrame] = {}
    if "persist_observed" in baselines:
        predictions["persist_observed"] = predict_persist_observed(checkpoint_rows)

    frozen_matrix: pd.DataFrame | None = None
    if preseason is not None and any(b in _BASELINES_NEED_PRESEASON for b in baselines):
        frozen_matrix = _preseason_rate_matrix(checkpoint_rows, preseason)

    if "frozen_preseason" in baselines and frozen_matrix is not None:
        predictions["frozen_preseason"] = frozen_matrix
    if "marcel_blend" in baselines and frozen_matrix is not None:
        blended = predict_marcel_blend(
            checkpoint_rows, preseason, prior_pa=prior_pa,
            preseason_matrix=frozen_matrix,
        )
        if blended is not None:
            predictions["marcel_blend"] = blended

    if not predictions:
        return {"n_players": 0, "systems": {}}

    y_true = _ros_target_matrix(checkpoint_rows)

    # Sample filter: require non-NaN targets + sufficient ROS PA + non-NaN predictions
    valid = y_true.notna().all(axis=1)
    if "ros_pa" in checkpoint_rows.columns:
        valid = valid & (checkpoint_rows["ros_pa"].reset_index(drop=True) >= min_ros_pa)
    for name, pred in predictions.items():
        valid = valid & pred.notna().all(axis=1).reset_index(drop=True)

    n = int(valid.sum())
    if n == 0:
        return {"n_players": 0, "systems": {}}

    y_true_f = y_true.loc[valid].reset_index(drop=True)
    systems: dict[str, dict] = {}
    for name, pred in predictions.items():
        pred_f = pred.loc[valid].reset_index(drop=True)
        systems[name] = {
            "y_pred": pred_f.values,
            "metrics": compute_metrics(
                y_true_f, pred_f, list(ROS_TARGET_DISPLAY),
            ),
        }

    return {"n_players": n, "y_true": y_true_f.values, "systems": systems}


def evaluate_year(
    year: int,
    snapshots: pd.DataFrame,
    thresholds: list[int],
    baselines: list[str],
    preseason: pd.DataFrame | None,
    prior_pa: float,
    min_ros_pa: int,
) -> dict:
    """Run all thresholds for one season's snapshots."""
    yearly = snapshots[snapshots["season"] == year]
    checkpoints = pa_checkpoint_rows(yearly, thresholds=thresholds)
    out: dict = {"year": year, "thresholds": {}}
    for t, rows in checkpoints.items():
        result = evaluate_checkpoint(
            rows, baselines, preseason, prior_pa, min_ros_pa,
        )
        out["thresholds"][t] = result
        sys_summary = ", ".join(
            f"{_BASELINE_DISPLAY.get(k, k)}={v['metrics']['aggregate']['rmse']:.4f}"
            for k, v in result["systems"].items()
        ) or "(none)"
        logger.info(
            "  %d @ %d PA: n=%d | %s",
            year, t, result["n_players"], sys_summary,
        )
    return out


# ---------------------------------------------------------------------------
# Pooling & reporting
# ---------------------------------------------------------------------------


def pool_by_threshold(year_results: list[dict]) -> dict:
    """Concatenate y_true/y_pred across years at each threshold.

    Returns ``{threshold: {"n_players": int, "systems": {name: metrics}}}``.
    """
    thresholds: set[int] = set()
    for yr in year_results:
        thresholds.update(yr["thresholds"].keys())

    pooled: dict[int, dict] = {}
    for t in sorted(thresholds):
        system_names: set[str] = set()
        for yr in year_results:
            system_names.update(yr["thresholds"].get(t, {}).get("systems", {}).keys())

        per_system: dict[str, dict] = {}
        n_players_accum: dict[str, int] = {}
        for name in system_names:
            y_trues, y_preds = [], []
            for yr in year_results:
                cell = yr["thresholds"].get(t, {})
                if name in cell.get("systems", {}):
                    y_trues.append(cell["y_true"])
                    y_preds.append(cell["systems"][name]["y_pred"])
            if y_trues:
                yt = np.concatenate(y_trues, axis=0)
                yp = np.concatenate(y_preds, axis=0)
                per_system[name] = compute_metrics(
                    yt, yp, list(ROS_TARGET_DISPLAY),
                )
                n_players_accum[name] = int(yt.shape[0])

        pooled[t] = {
            "systems": per_system,
            "n_players_per_system": n_players_accum,
        }
    return pooled


def print_benchmark_results(
    year_results: list[dict],
    pooled: dict[int, dict],
) -> None:
    if not year_results or not pooled:
        print("\nNo ROS benchmark results to display.")
        return

    thresholds = sorted(pooled)
    system_order = _preferred_system_order(pooled)

    print("\n" + "=" * 80)
    print(f"  ROS Benchmark: PA checkpoints {thresholds}")
    years = sorted({yr["year"] for yr in year_results})
    print(f"  Years evaluated: {years}")
    print("=" * 80)

    for t in thresholds:
        cell = pooled[t]
        systems = cell["systems"]
        if not systems:
            continue
        n_by_sys = cell["n_players_per_system"]
        print(f"\n  [{t} PA checkpoint]  n per system: "
              + ", ".join(f"{_BASELINE_DISPLAY.get(s, s)}={n_by_sys[s]}" for s in systems))
        print(f"  {'System':<14}", end="")
        for tgt in ROS_TARGET_DISPLAY:
            print(f"  {tgt:>8}", end="")
        print(f"  {'Mean':>8}")
        print(f"  {'-' * (14 + 10 * (len(ROS_TARGET_DISPLAY) + 1))}")
        for name in system_order:
            if name not in systems:
                continue
            m = systems[name]
            print(f"  {_BASELINE_DISPLAY.get(name, name):<14}", end="")
            for tgt in ROS_TARGET_DISPLAY:
                print(f"  {m['per_target'][tgt]['rmse']:>8.4f}", end="")
            print(f"  {m['aggregate']['rmse']:>8.4f}")
    print()


def _preferred_system_order(pooled: dict[int, dict]) -> list[str]:
    all_systems: set[str] = set()
    for cell in pooled.values():
        all_systems.update(cell["systems"].keys())
    ordered = [s for s in ALL_BASELINES if s in all_systems]
    for s in sorted(all_systems):
        if s not in ordered:
            ordered.append(s)
    return ordered


def save_benchmark_outputs(
    year_results: list[dict],
    pooled: dict[int, dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_report: dict = {
        "years": sorted({yr["year"] for yr in year_results}),
        "thresholds": sorted(pooled),
        "pooled": {},
        "per_year": [],
    }
    system_order = _preferred_system_order(pooled)
    for t, cell in pooled.items():
        json_report["pooled"][str(t)] = {
            "n_players_per_system": cell["n_players_per_system"],
            "systems": {
                name: cell["systems"][name] for name in system_order
                if name in cell["systems"]
            },
        }
    for yr in year_results:
        yr_entry: dict = {"year": yr["year"], "thresholds": {}}
        for t, cell in yr["thresholds"].items():
            yr_entry["thresholds"][str(t)] = {
                "n_players": cell.get("n_players", 0),
                "systems": {
                    name: cell["systems"][name]["metrics"]
                    for name in cell.get("systems", {})
                },
            }
        json_report["per_year"].append(yr_entry)

    json_path = output_dir / "benchmark_ros_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    logger.info("Saved JSON report → %s", json_path)

    rows: list[dict] = []
    for t in sorted(pooled):
        cell = pooled[t]
        for name in system_order:
            metrics = cell["systems"].get(name)
            if metrics is None:
                continue
            rows.append({
                "threshold": t,
                "system": name,
                "n_players": cell["n_players_per_system"].get(name, 0),
                "target": "Mean",
                "rmse": metrics["aggregate"]["rmse"],
                "mae": metrics["aggregate"]["mae"],
                "r2": metrics["aggregate"]["r2"],
                "mape": metrics["aggregate"]["mape"],
            })
            for tgt in ROS_TARGET_DISPLAY:
                pt = metrics["per_target"][tgt]
                rows.append({
                    "threshold": t,
                    "system": name,
                    "n_players": cell["n_players_per_system"].get(name, 0),
                    "target": tgt,
                    "rmse": pt["rmse"],
                    "mae": pt["mae"],
                    "r2": pt["r2"],
                    "mape": pt["mape"],
                })
    csv_path = output_dir / "benchmark_ros_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Saved CSV table → %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROS benchmark: evaluate weekly baselines at PA checkpoints.",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2023, 2024, 2025],
        help="Evaluation years (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--thresholds", type=int, nargs="+",
        default=list(DEFAULT_PA_CHECKPOINTS),
        help=f"PA checkpoints to evaluate (default: {list(DEFAULT_PA_CHECKPOINTS)})",
    )
    parser.add_argument(
        "--include", nargs="+", choices=list(ALL_BASELINES), default=list(ALL_BASELINES),
        help=f"Baselines to include (default: all {list(ALL_BASELINES)})",
    )
    parser.add_argument(
        "--prior-pa", type=float, default=200.0,
        help="Prior PA weight for marcel_blend (default: 200)",
    )
    parser.add_argument(
        "--min-ros-pa", type=int, default=50,
        help="Minimum ros_pa required for a row to count (default: 50)",
    )
    parser.add_argument(
        "--raw-dir", default="data/raw",
        help="Directory containing weekly_snapshots_{year}.parquet files",
    )
    parser.add_argument(
        "--cache-dir", default=str(_DEFAULT_CACHE_DIR),
        help="Directory for per-year preseason MTL prediction cache",
    )
    parser.add_argument(
        "--output-dir", default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory for report JSON + CSV",
    )
    parser.add_argument(
        "--data", default="data/merged_batter_data.parquet",
        help="Merged dataset for preseason retraining (only used with --retrain)",
    )
    parser.add_argument(
        "--data-config", default="configs/data.yaml",
        help="Data config YAML (only used with --retrain)",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Retrain preseason MTL for each year when cache is missing (slow).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    years = sorted(args.years)

    logger.info(
        "Loading weekly snapshots for %s from %s …",
        years, args.raw_dir,
    )
    snapshots = load_weekly_snapshots(years, raw_dir=args.raw_dir)

    df_featured: pd.DataFrame | None = None
    data_config: dict | None = None
    need_preseason = any(b in args.include for b in _BASELINES_NEED_PRESEASON)
    if need_preseason and args.retrain:
        logger.info("Loading merged data + building features (for retraining) …")
        df = pd.read_parquet(args.data)
        with open(args.data_config) as f:
            data_config = yaml.safe_load(f)
        df_featured = build_features(df, data_config)

    year_results: list[dict] = []
    for year in years:
        logger.info("=" * 60)
        logger.info("Year %d", year)
        logger.info("=" * 60)
        preseason = None
        if need_preseason:
            preseason = load_or_generate_preseason_cache(
                year,
                cache_dir=cache_dir,
                df_featured=df_featured,
                data_config=data_config,
                retrain=args.retrain,
                seed=args.seed,
            )
        result = evaluate_year(
            year=year,
            snapshots=snapshots,
            thresholds=args.thresholds,
            baselines=args.include,
            preseason=preseason,
            prior_pa=args.prior_pa,
            min_ros_pa=args.min_ros_pa,
        )
        year_results.append(result)

    pooled = pool_by_threshold(year_results)
    print_benchmark_results(year_results, pooled)
    save_benchmark_outputs(year_results, pooled, output_dir)

    logger.info("ROS benchmark complete.")


if __name__ == "__main__":
    main()

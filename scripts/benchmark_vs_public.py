"""Benchmark our models vs external projections (Steamer, ZiPS) across 2022-2025.

Rolling retraining ensures each evaluation year is truly out-of-sample. All
systems are evaluated on the same player set using actual outcomes as ground
truth.  Metrics are pooled (concatenated) across years for statistical
robustness.

Usage
-----
    uv run python scripts/benchmark_vs_public.py                         # Full benchmark
    uv run python scripts/benchmark_vs_public.py --years 2024 2025       # Specific years
    uv run python scripts/benchmark_vs_public.py --systems steamer zips    # Specific public systems
    uv run python scripts/benchmark_vs_public.py --no-retrain            # Saved models, latest year only
    uv run python scripts/benchmark_vs_public.py --with-plots            # Generate comparison plots
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import get_production_data
from src.eval.metrics import compute_metrics
from src.eval.pa_projection import project_pa
from src.eval.plots import save_figure
from src.features.pipeline import build_features, extract_xy
from src.features.registry import COUNT_STATS, RATE_STATS, TARGET_COLUMNS, TARGET_DISPLAY, TARGET_STATS
from src.models.mtl.model import MTLEnsembleForecaster, MTLForecaster
from src.models.utils import align_features, get_model_configs

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = get_model_configs()

_SYSTEM_DISPLAY = {
    "steamer": "Steamer",
    "zips": "ZiPS",
}

# Ordered: our models first, then external, then naive
_SYSTEM_ORDER_KEYS = ["MTL", "Steamer", "ZiPS", "Naive"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ordered_systems(available: set[str]) -> list[str]:
    """Return system names in canonical display order."""
    ordered = [s for s in _SYSTEM_ORDER_KEYS if s in available]
    # Append any unexpected systems alphabetically
    for s in sorted(available):
        if s not in ordered:
            ordered.insert(-1, s)  # before Naive
    return ordered


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

_KEEP_STAT_COLS = ["obp", "slg", "hr", "r", "rbi", "sb", "pa"]

_EXT_CACHE_DIR = Path("data/reports/benchmark/cache")


def _load_single_ext_projection(
    system: str,
    year: int,
    ext_dir: Path,
) -> pd.DataFrame | None:
    """Load and normalize one external projection CSV, with file-based cache.

    The cache is a parquet file that is invalidated when the source CSV is
    newer than the cached file.
    """
    path = ext_dir / f"{system}_{year}.csv"
    if not path.exists():
        logger.warning("Missing projection file: %s", path)
        return None

    # Check cache
    cache_path = _EXT_CACHE_DIR / f"{system}_{year}.parquet"
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        df = pd.read_parquet(cache_path)
        logger.info("  Loaded %s %d from cache: %d players", system, year, len(df))
        return df

    # Cache miss — parse and normalize CSV
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    rename_map = {}
    if "playerid" in df.columns:
        rename_map["playerid"] = "idfg"
    if "playername" in df.columns:
        rename_map["playername"] = "name"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "idfg" not in df.columns:
        logger.warning("No player ID column in %s — skipping", path)
        return None

    df["idfg"] = pd.to_numeric(df["idfg"], errors="coerce")
    df = df.dropna(subset=["idfg"])
    df["idfg"] = df["idfg"].astype(int)

    for col in _KEEP_STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["idfg"], keep="first")
    keep = ["idfg"] + [c for c in _KEEP_STAT_COLS if c in df.columns]
    df = df[keep].copy()
    df = df.set_index("idfg")

    # Save cache
    _EXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, engine="pyarrow", compression="zstd")
    logger.info("  Loaded %s %d: %d players (cached)", system, year, len(df))
    return df


def _load_external_projections(
    year: int,
    systems: list[str],
    ext_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load and normalize external projection CSVs for one year.

    Handles both raw FanGraphs format (Name, PlayerId, OBP, ...) and
    pre-normalized format (idfg, name, obp, ...).  Results are cached
    as parquet files so subsequent runs skip CSV parsing.

    Returns a dict mapping system name → DataFrame indexed by integer idfg.
    """
    ext_dir = Path(ext_dir)
    result: dict[str, pd.DataFrame] = {}

    for system in systems:
        df = _load_single_ext_projection(system, year, ext_dir)
        if df is not None:
            result[system] = df

    return result


def train_model_for_year(
    model_key: str,
    retrain_df: pd.DataFrame,
    data_config: dict,
    seed: int = 42,
    device: str = "cpu",
) -> object:
    """Train a fresh model on the retrain set for one evaluation fold.

    The most recent year in retrain_df is carved out as eval_set for early stopping.
    """
    if retrain_df["season"].nunique() > 1:
        max_season = retrain_df["season"].max()
        train_part = retrain_df[retrain_df["season"] != max_season]
        eval_part = retrain_df[retrain_df["season"] == max_season]
        X_train, y_train = extract_xy(train_part, data_config)
        X_eval, y_eval = extract_xy(eval_part, data_config)
        eval_set: tuple | None = (X_eval, y_eval)
    else:
        train_part = retrain_df
        X_train, y_train = extract_xy(retrain_df, data_config)
        eval_set = None

    season = train_part["season"].values if "season" in train_part.columns else None

    info = MODEL_CONFIGS[model_key]
    with open(info["config_path"]) as f:
        config = yaml.safe_load(f)
    config["seed"] = seed

    ensemble_cfg = config.get("ensemble", {})
    ensemble_n_seeds = ensemble_cfg.get("n_seeds", 0)
    if ensemble_n_seeds > 1:
        model = MTLEnsembleForecaster(config)
    else:
        model = MTLForecaster(config)
    model.fit(X_train, y_train, eval_set=eval_set, season=season)

    return model


def _load_saved_model(model_key: str) -> object:
    """Load a pre-trained model from disk."""
    info = MODEL_CONFIGS[model_key]
    model_dir = Path(info["model_dir"])
    if model_key == "mtl":
        ensemble_path = model_dir / "ensemble_meta.json"
        if ensemble_path.exists():
            return MTLEnsembleForecaster.load(model_dir)
    return info["class"].load(model_dir)


def evaluate_year(
    eval_year: int,
    df_featured: pd.DataFrame,
    data_config: dict,
    systems: list[str],
    ext_dir: str | Path,
    min_pa: int,
    seed: int,
    no_retrain: bool = False,
    device: str = "cpu",
) -> dict | None:
    """Run full evaluation for one year.

    Returns a dict with year, n_players, y_true, and per-system y_pred + metrics,
    or None if the year cannot be evaluated.
    """
    logger.info("=" * 60)
    logger.info("Evaluating year %d", eval_year)
    logger.info("=" * 60)

    # Production split: train on data through eval_year-2, predict from eval_year-1
    retrain_df, predict_df = get_production_data(
        df_featured, end_year=eval_year - 1, target_cols=TARGET_COLUMNS,
    )

    # Filter predict set to players with valid targets and sufficient actual PA
    predict_df = predict_df.dropna(subset=TARGET_COLUMNS)
    if "target_pa" in predict_df.columns:
        predict_df = predict_df[predict_df["target_pa"] >= min_pa].copy()

    logger.info(
        "  Retrain: %d rows | Test: %d players (PA >= %d in %d)",
        len(retrain_df), len(predict_df), min_pa, eval_year,
    )

    if len(predict_df) == 0:
        logger.warning("  No test players for %d — skipping", eval_year)
        return None

    # Ground truth and predict features
    X_predict, y_true = extract_xy(predict_df, data_config)

    # --- Rate-to-count conversion for evaluation ---
    # When rate_targets is enabled, y_true and model predictions are per-PA rates.
    # Convert to counts for apples-to-apples comparison with external projections.
    # Use Marcel-projected PA (not actual PA) for model predictions so that the
    # benchmark is fair — external projections must also predict playing time.
    # y_true uses actual PA since those are ground-truth outcomes.
    rate_targets = data_config.get("rate_targets", False)
    projected_pa: pd.Series | None = None
    if rate_targets and "target_pa" in predict_df.columns:
        actual_pa = predict_df["target_pa"].astype(float)
        # Convert y_true from rates to actual counts (ground truth)
        for col in y_true.columns:
            stat = col.replace("target_", "")
            if stat in COUNT_STATS:
                y_true[col] = y_true[col] * actual_pa.values
        # Project PA using Marcel formula for model predictions (fair comparison)
        season_games = data_config.get("season_games", {})
        projected_pa = project_pa(predict_df, season_games)

    # --- Naive baseline: predict year Y = year Y-1 stats ---
    naive_raw = predict_df[TARGET_STATS].copy()
    naive_pred = naive_raw.rename(columns={s: f"target_{s}" for s in TARGET_STATS})

    # --- Train / load MTL and predict ---
    mtl_display = MODEL_CONFIGS["mtl"]["display_name"]
    mtl_preds: pd.DataFrame | None = None
    try:
        if no_retrain:
            logger.info("  Loading saved MTL …")
            model = _load_saved_model("mtl")
        else:
            logger.info("  Training MTL for %d …", eval_year)
            model = train_model_for_year(
                "mtl", retrain_df, data_config, seed, device,
            )
        X_aligned = align_features(X_predict, model, mtl_display)
        mtl_preds = model.predict(X_aligned)
        # Convert rate predictions to counts using projected PA
        if rate_targets and projected_pa is not None:
            for col in mtl_preds.columns:
                stat = col.replace("target_", "")
                if stat in COUNT_STATS:
                    mtl_preds[col] = mtl_preds[col] * projected_pa.values
    except Exception:
        logger.exception("  Failed MTL for %d", eval_year)

    # --- Load external projections ---
    ext_preds: dict[str, pd.DataFrame] = {}
    ext_loaded = _load_external_projections(eval_year, systems, ext_dir)
    if ext_loaded:
        predict_idfg = predict_df["idfg"].astype(int)
        for system, sys_lookup in ext_loaded.items():
            mapped = pd.DataFrame(index=predict_df.index)
            for stat in TARGET_STATS:
                if stat in sys_lookup.columns:
                    mapped[f"target_{stat}"] = predict_idfg.map(
                        sys_lookup[stat]
                    ).values
            ext_preds[system] = mapped
    else:
        logger.warning("  No external projections found for %d", eval_year)

    # --- Build player intersection (all systems must have values) ---
    valid_mask = naive_pred.notna().all(axis=1)
    for preds in ext_preds.values():
        valid_mask = valid_mask & preds.notna().all(axis=1)
    if mtl_preds is not None:
        preds_notna = mtl_preds.notna().all(axis=1)
        preds_notna.index = valid_mask.index
        valid_mask = valid_mask & preds_notna

    n_valid = int(valid_mask.sum())
    logger.info("  Player intersection: %d / %d", n_valid, len(predict_df))

    if n_valid == 0:
        logger.warning("  Empty intersection for %d — skipping", eval_year)
        return None

    # Filter everything to the intersection
    y_true_f = y_true.loc[valid_mask].reset_index(drop=True)
    y_true_arr = y_true_f.values

    result: dict = {
        "year": eval_year,
        "n_players": n_valid,
        "y_true": y_true_arr,
        "systems": {},
    }

    # Naive
    naive_f = naive_pred.loc[valid_mask].reset_index(drop=True)
    result["systems"]["Naive"] = {
        "y_pred": naive_f.values,
        "metrics": compute_metrics(y_true_f, naive_f, TARGET_DISPLAY),
    }

    # MTL
    if mtl_preds is not None:
        preds_f = mtl_preds.loc[valid_mask].reset_index(drop=True)
        result["systems"][mtl_display] = {
            "y_pred": preds_f.values,
            "metrics": compute_metrics(y_true_f, preds_f, TARGET_DISPLAY),
        }

    # External projections
    for system, preds in ext_preds.items():
        display = _SYSTEM_DISPLAY.get(system, system.capitalize())
        preds_f = preds.loc[valid_mask].reset_index(drop=True)
        result["systems"][display] = {
            "y_pred": preds_f.values,
            "metrics": compute_metrics(y_true_f, preds_f, TARGET_DISPLAY),
        }

    # Log per-year summary
    for sys_name in _ordered_systems(set(result["systems"])):
        agg_rmse = result["systems"][sys_name]["metrics"]["aggregate"]["rmse"]
        logger.info("  %s mean RMSE: %.4f", sys_name, agg_rmse)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(year_results: list[dict | None]) -> dict[str, dict]:
    """Pool y_true/y_pred across years and compute pooled metrics."""
    valid = [r for r in year_results if r is not None]
    if not valid:
        return {}

    all_system_names: set[str] = set()
    for yr in valid:
        all_system_names.update(yr["systems"].keys())

    pooled: dict[str, dict] = {}
    for sys_name in all_system_names:
        y_trues, y_preds = [], []
        for yr in valid:
            if sys_name in yr["systems"]:
                y_trues.append(yr["y_true"])
                y_preds.append(yr["systems"][sys_name]["y_pred"])
        if y_trues:
            pooled_true = np.concatenate(y_trues, axis=0)
            pooled_pred = np.concatenate(y_preds, axis=0)
            pooled[sys_name] = compute_metrics(pooled_true, pooled_pred, TARGET_DISPLAY)

    return pooled


# ---------------------------------------------------------------------------
# Output: console
# ---------------------------------------------------------------------------


def print_benchmark_results(
    year_results: list[dict | None],
    pooled: dict[str, dict],
) -> None:
    """Print benchmark tables to console."""
    valid = [r for r in year_results if r is not None]
    if not valid or not pooled:
        print("\nNo valid results to display.")
        return

    system_order = _ordered_systems(set(pooled))
    years = [r["year"] for r in valid]
    total_players = sum(r["n_players"] for r in valid)

    print(f"\n{'=' * 80}")
    print(f"  Benchmark: Our Models vs External Projections ({min(years)}-{max(years)})")
    print(f"  {len(years)} evaluation years, {total_players} total player-seasons")
    print(f"{'=' * 80}")

    # Per-year player counts
    for r in valid:
        print(f"    {r['year']}: {r['n_players']} players")

    # --- Pooled RMSE table ---
    print(f"\n  Pooled RMSE (lower is better)")
    _print_metric_table(pooled, system_order, "rmse")

    # --- Pooled R² table ---
    print(f"\n  Pooled R² (higher is better)")
    _print_metric_table(pooled, system_order, "r2", fmt_rate=".4f", fmt_count=".3f")

    # --- Per-year RMSE ---
    print(f"\n  Per-Year Mean RMSE")
    print(f"  {'System':<12}", end="")
    for yr in valid:
        print(f"  {yr['year']:>8}", end="")
    print()
    print(f"  {'-' * (12 + 10 * len(valid))}")
    for sys_name in system_order:
        print(f"  {sys_name:<12}", end="")
        for yr in valid:
            if sys_name in yr["systems"]:
                val = yr["systems"][sys_name]["metrics"]["aggregate"]["rmse"]
                print(f"  {val:>8.4f}", end="")
            else:
                print(f"  {'—':>8}", end="")
        print()

    # --- Win counts (per-target lowest RMSE) ---
    print(f"\n  Target Wins (lowest pooled RMSE per target)")
    wins: dict[str, int] = {s: 0 for s in system_order}
    for target in TARGET_DISPLAY:
        best_sys = min(
            system_order,
            key=lambda s: pooled[s]["per_target"][target]["rmse"],
        )
        wins[best_sys] += 1
    for sys_name in system_order:
        marker = " *" if wins[sys_name] == max(wins.values()) else ""
        print(f"    {sys_name:<12} {wins[sys_name]}/{len(TARGET_DISPLAY)}{marker}")

    # --- Per-year ranking tables ---
    _print_year_rankings(valid)

    print()


def _print_metric_table(
    pooled: dict[str, dict],
    system_order: list[str],
    metric: str,
    fmt_rate: str = ".4f",
    fmt_count: str = ".2f",
) -> None:
    """Print a per-target metric table."""
    print(f"  {'System':<12}", end="")
    for t in TARGET_DISPLAY:
        print(f"  {t:>8}", end="")
    print(f"  {'Mean':>8}")
    print(f"  {'-' * (12 + 10 * (len(TARGET_DISPLAY) + 1))}")

    for sys_name in system_order:
        metrics = pooled[sys_name]
        print(f"  {sys_name:<12}", end="")
        for t in TARGET_DISPLAY:
            val = metrics["per_target"][t][metric]
            fmt = fmt_rate if t.lower() in RATE_STATS else fmt_count
            print(f"  {val:>8{fmt}}", end="")
        agg = metrics["aggregate"][metric]
        print(f"  {agg:>8.4f}")


def _rank_systems_for_year(year_result: dict) -> list[dict]:
    """Compute per-target RMSE ranks for one year.

    Returns a list of dicts, one per system, with keys:
    ``system``, one key per target in ``TARGET_DISPLAY``, and ``avg_rank``.
    Ranks use min-rank convention for ties (both get 1, next gets 3).
    """
    systems = year_result["systems"]
    system_order = _ordered_systems(set(systems))

    # Collect RMSE values: {target: [(system, rmse), ...]}
    target_rmses: dict[str, list[tuple[str, float]]] = {}
    for target in TARGET_DISPLAY:
        vals = []
        for sys_name in system_order:
            rmse_val = systems[sys_name]["metrics"]["per_target"][target]["rmse"]
            vals.append((sys_name, rmse_val))
        target_rmses[target] = vals

    # Rank per target (ascending RMSE = rank 1)
    ranks: dict[str, dict[str, int]] = {s: {} for s in system_order}
    for target, vals in target_rmses.items():
        sorted_vals = sorted(vals, key=lambda x: x[1])
        rank = 1
        for i, (sys_name, rmse_val) in enumerate(sorted_vals):
            if i > 0 and rmse_val > sorted_vals[i - 1][1]:
                rank = i + 1
            ranks[sys_name][target] = rank

    # Build output rows
    rows = []
    for sys_name in system_order:
        row: dict = {"system": sys_name}
        target_ranks = []
        for target in TARGET_DISPLAY:
            r = ranks[sys_name][target]
            row[target] = r
            target_ranks.append(r)
        row["avg_rank"] = sum(target_ranks) / len(target_ranks)
        rows.append(row)

    # Sort by avg_rank for display
    rows.sort(key=lambda r: r["avg_rank"])
    return rows


def _print_year_rankings(valid_results: list[dict]) -> None:
    """Print a ranking table for each evaluation year."""
    for yr in valid_results:
        rows = _rank_systems_for_year(yr)
        print(f"\n  {yr['year']} Rankings (by RMSE, 1 = best, {yr['n_players']} players)")
        print(f"  {'System':<12}", end="")
        for t in TARGET_DISPLAY:
            print(f"  {t:>5}", end="")
        print(f"  {'Avg':>6}")
        print(f"  {'-' * (12 + 7 * len(TARGET_DISPLAY) + 8)}")
        for row in rows:
            print(f"  {row['system']:<12}", end="")
            for t in TARGET_DISPLAY:
                print(f"  {row[t]:>5}", end="")
            print(f"  {row['avg_rank']:>6.1f}")


# ---------------------------------------------------------------------------
# Output: files
# ---------------------------------------------------------------------------


def save_benchmark_outputs(
    year_results: list[dict | None],
    pooled: dict[str, dict],
    output_dir: str | Path,
) -> None:
    """Save benchmark report (JSON) and summary table (CSV)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    valid = [r for r in year_results if r is not None]
    system_order = _ordered_systems(set(pooled))

    # --- JSON report ---
    report: dict = {
        "years": [r["year"] for r in valid],
        "total_player_seasons": sum(r["n_players"] for r in valid),
        "pooled": {},
        "per_year": [],
    }
    for sys_name in system_order:
        if sys_name in pooled:
            report["pooled"][sys_name] = pooled[sys_name]
    for yr in valid:
        yr_entry: dict = {"year": yr["year"], "n_players": yr["n_players"], "systems": {}}
        for sys_name in system_order:
            if sys_name in yr["systems"]:
                yr_entry["systems"][sys_name] = yr["systems"][sys_name]["metrics"]
        report["per_year"].append(yr_entry)

    json_path = out / "benchmark_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report → %s", json_path)

    # --- CSV table ---
    rows = []
    for sys_name in system_order:
        if sys_name not in pooled:
            continue
        metrics = pooled[sys_name]
        for target in TARGET_DISPLAY:
            pt = metrics["per_target"][target]
            rows.append({
                "system": sys_name,
                "target": target,
                "rmse": pt["rmse"],
                "mae": pt["mae"],
                "r2": pt["r2"],
                "mape": pt["mape"],
            })
        agg = metrics["aggregate"]
        rows.append({
            "system": sys_name,
            "target": "Mean",
            "rmse": agg["rmse"],
            "mae": agg["mae"],
            "r2": agg["r2"],
            "mape": agg["mape"],
        })

    csv_path = out / "benchmark_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Saved table → %s", csv_path)

    # --- Rankings CSV ---
    ranking_rows = []
    for yr in valid:
        for row in _rank_systems_for_year(yr):
            ranking_rows.append({"year": yr["year"], **row})
    if ranking_rows:
        rankings_path = out / "benchmark_rankings.csv"
        pd.DataFrame(ranking_rows).to_csv(rankings_path, index=False)
        logger.info("Saved rankings → %s", rankings_path)


# ---------------------------------------------------------------------------
# Output: plots
# ---------------------------------------------------------------------------


_PLOT_COLORS = {
    "MTL": "#DD8452",
    "Steamer": "#C44E52",
    "ZiPS": "#8172B3",
    "Naive": "#CCCCCC",
}


def generate_benchmark_plots(
    pooled: dict[str, dict],
    output_dir: str | Path,
) -> None:
    """Generate grouped bar chart comparing RMSE across systems and targets."""
    if not pooled:
        return

    system_order = _ordered_systems(set(pooled))
    labels = TARGET_DISPLAY + ["Mean"]
    n_groups = len(labels)
    n_systems = len(system_order)

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 2), 6))
    x = np.arange(n_groups)
    width = 0.7 / n_systems

    for j, sys_name in enumerate(system_order):
        metrics = pooled[sys_name]
        values = [metrics["per_target"][t]["rmse"] for t in TARGET_DISPLAY]
        values.append(metrics["aggregate"]["rmse"])

        color = _PLOT_COLORS.get(sys_name, f"C{j}")
        offset = (j - n_systems / 2 + 0.5) * width
        alpha = 0.7 if sys_name == "Naive" else 0.85
        kwargs = {}
        if sys_name == "Naive":
            kwargs = {"hatch": "//", "edgecolor": "gray"}
        ax.bar(x + offset, values, width, label=sys_name, color=color, alpha=alpha, **kwargs)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RMSE")
    ax.set_title("Benchmark: Pooled RMSE by Target")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(output_dir) / "benchmark_rmse.png"
    save_figure(fig, out_path)
    logger.info("Saved plot → %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark our models vs external projections (2022-2025)",
    )
    parser.add_argument(
        "--data", default="data/merged_batter_data.parquet",
        help="Path to merged Parquet file",
    )
    parser.add_argument(
        "--data-config", default="configs/data.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--ext-dir", default="data/external_projections",
        help="Directory containing external projection CSVs",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025],
        help="Evaluation years (default: 2022 2023 2024 2025)",
    )
    parser.add_argument(
        "--systems", nargs="+", default=["steamer", "zips"],
        help="External projection systems to compare (default: steamer zips)",
    )
    parser.add_argument(
        "--min-pa", type=int, default=200,
        help="Minimum actual PA in ground truth year for inclusion (default: 200)",
    )
    parser.add_argument(
        "--no-retrain", action="store_true",
        help="Skip retraining; load saved models and evaluate latest year only",
    )
    parser.add_argument(
        "--with-plots", action="store_true",
        help="Generate comparison plots",
    )
    parser.add_argument(
        "--output-dir", default="data/reports/benchmark/",
        help="Output directory for reports and plots",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for MTL training: cpu, cuda, or mps (default: cpu)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for model training",
    )
    args = parser.parse_args()

    # If --no-retrain, only evaluate the latest year with saved models
    if args.no_retrain:
        args.years = [max(args.years)]
        logger.info("--no-retrain: evaluating year %d only with saved models", args.years[0])

    # Load data and config
    logger.info("Loading merged data from %s", args.data)
    df = pd.read_parquet(args.data)
    with open(args.data_config) as f:
        data_config = yaml.safe_load(f)

    # Build features once for all years
    logger.info("Building features for %d rows …", len(df))
    df_featured = build_features(df, data_config)

    # Run per-year evaluations
    year_results: list[dict | None] = []
    for year in sorted(args.years):
        result = evaluate_year(
            eval_year=year,
            df_featured=df_featured,
            data_config=data_config,
            systems=args.systems,
            ext_dir=args.ext_dir,
            min_pa=args.min_pa,
            seed=args.seed,
            no_retrain=args.no_retrain,
            device=args.device,
        )
        year_results.append(result)

    # Aggregate across years
    pooled = aggregate_results(year_results)

    # Output
    print_benchmark_results(year_results, pooled)
    save_benchmark_outputs(year_results, pooled, args.output_dir)

    if args.with_plots:
        generate_benchmark_plots(pooled, args.output_dir)

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()

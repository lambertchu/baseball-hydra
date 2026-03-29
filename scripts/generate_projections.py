"""Project 2026 season stats for MLB batters.

Loads the merged dataset, builds features, and generates projections using
the MTL model. By default loads a pre-trained model; use --retrain to train
from scratch on all historical data.

Usage
-----
    uv run python scripts/generate_projections.py                    # Load saved MTL model
    uv run python scripts/generate_projections.py --retrain          # Retrain from scratch
    uv run python scripts/generate_projections.py --with-public      # Compare with public projections
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetch_projections import (
    DISPLAY_NAMES as PROJ_DISPLAY_NAMES,
    PROJECTION_SYSTEMS,
    fetch_all_projections,
    load_projections,
)
from src.data.splits import get_production_data
from src.eval.metrics import compute_metrics, compute_naive_baseline
from src.eval.pa_projection import project_pa, rate_to_count
from src.eval.report import print_report
from src.features.pipeline import build_features, extract_xy
from src.features.registry import COUNT_STATS, RATE_STATS, TARGET_COLUMNS, TARGET_DISPLAY, TARGET_STATS
from src.models.mtl.model import MTLEnsembleForecaster, MTLForecaster
from src.models.utils import align_features, get_model_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_PROJECTIONS_DIR = Path("data/projections")
MODEL_CONFIGS = get_model_configs()

# Domain-informed upper bounds for count stats (historical MLB extremes + margin).
_COUNT_UPPER_BOUNDS: dict[str, int] = {"HR": 80, "R": 175, "RBI": 175, "SB": 120}


def _load_or_train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    retrain: bool,
    seed: int = 42,
    eval_set: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    season: np.ndarray | None = None,
) -> object:
    """Load a pre-trained MTL model or retrain from scratch."""
    info = MODEL_CONFIGS["mtl"]
    model_dir = Path(info["model_dir"])

    if not retrain and model_dir.exists():
        logger.info("Loading pre-trained MTL from %s", model_dir)
        ensemble_meta = model_dir / "ensemble_meta.json"
        if ensemble_meta.exists():
            return MTLEnsembleForecaster.load(model_dir)
        return MTLForecaster.load(model_dir)

    logger.info("Training MTL on all historical data …")
    with open(info["config_path"]) as f:
        config = yaml.safe_load(f)
    config["seed"] = seed

    ensemble_cfg = config.get("ensemble", {})
    if ensemble_cfg.get("n_seeds", 0) > 1:
        model = MTLEnsembleForecaster(config)
    else:
        model = MTLForecaster(config)

    model.fit(X_train, y_train, eval_set=eval_set, season=season)
    return model


def _build_single_model_results(
    predictions: pd.DataFrame,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Extract prediction columns with a model prefix."""
    result = {}
    for stat in TARGET_STATS:
        col = f"target_{stat}"
        result[f"{prefix} {stat.upper()}"] = predictions[col].values
    return result


def _round_and_clip(results: pd.DataFrame, prefix: str) -> None:
    """Round rate stats to 3dp and clip count stats to >= 0 in place."""
    for stat in RATE_STATS:
        col = f"{prefix} {stat.upper()}"
        if col in results.columns:
            results[col] = results[col].round(3)
    for stat in COUNT_STATS:
        col = f"{prefix} {stat.upper()}"
        if col in results.columns:
            upper = _COUNT_UPPER_BOUNDS.get(stat.upper(), 250)
            series = results[col].clip(lower=0, upper=upper).round(0)
            if series.isna().any():
                results[col] = series.astype("Int64")
            else:
                results[col] = series.astype(int)


def _load_external_projections(
    predict_year: int,
    systems: list[str],
    ext_dir: str | Path = "data/external_projections",
) -> pd.DataFrame | None:
    """Load cached external projection CSVs, concatenating available systems."""
    return load_projections(predict_year, systems=systems, out_dir=ext_dir)


def _merge_projections(
    results: pd.DataFrame,
    projections: pd.DataFrame,
) -> pd.DataFrame:
    """Pivot projections and join to results table on ``_idfg``.

    Expects ``results`` to already contain a ``_idfg`` column.
    Adds columns like ``"Stmr OBP"`` and ``"ZiPS HR"``.
    """
    results = results.copy()

    # Pivot from long (one row per player x system) to wide.
    systems_present = projections["projection_system"].unique()
    for system in systems_present:
        display = PROJ_DISPLAY_NAMES.get(system, system)
        sys_df = projections[projections["projection_system"] == system].copy()
        sys_df["idfg"] = sys_df["idfg"].astype(int)
        sys_df = sys_df.drop_duplicates(subset=["idfg"], keep="first")

        for stat in TARGET_STATS:
            if stat in sys_df.columns:
                col_name = f"{display} {stat.upper()}"
                mapping = sys_df.set_index("idfg")[stat]
                results[col_name] = results["_idfg"].map(mapping)

        _round_and_clip(results, display)

    return results


def _print_projection_comparison(
    results: pd.DataFrame,
    disp_prefix: str,
    predict_year: int,
    systems_present: list[str],
) -> None:
    """Print per-stat comparison blocks: our model vs public projections."""
    proj_displays = [PROJ_DISPLAY_NAMES.get(s, s) for s in systems_present]

    for stat in TARGET_STATS:
        upper = stat.upper()
        our_col = f"{disp_prefix} {upper}"
        if our_col not in results.columns:
            continue

        is_rate = stat in RATE_STATS

        # Build header.
        proj_headers = []
        for pd_name in proj_displays:
            col = f"{pd_name} {upper}"
            if col in results.columns:
                proj_headers.append((pd_name, col))

        width = 6 + 24 + 10 + 10 * len(proj_headers)
        print(f"\n{'=' * width}")
        print(f"  {predict_year} Projections vs Public Projections — {upper}")
        print(f"{'=' * width}")

        # Print header row.
        parts = [f"  {'#':>3}  {'Player':<24}  {disp_prefix:>8}"]
        for pd_name, _ in proj_headers:
            parts.append(f"  {pd_name:>8}")
        print("".join(parts))
        print(f"  {'-' * (width - 2)}")

        for i, (_, row) in enumerate(results.iterrows(), 1):
            val = row[our_col]
            parts = [f"  {i:>3}  {row['Player']:<24}"]
            if is_rate:
                parts.append(f"  {val:>8.3f}")
            else:
                parts.append(f"  {int(val):>8}")
            for _, col in proj_headers:
                pv = row.get(col)
                if pd.isna(pv):
                    parts.append(f"  {'—':>8}")
                elif is_rate:
                    parts.append(f"  {pv:>8.3f}")
                else:
                    parts.append(f"  {int(pv):>8}")
            print("".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 2026 season projections for MLB batters",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/merged_batter_data.parquet",
        help="Path to merged Parquet file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--min-pa",
        type=int,
        default=None,
        help="Minimum PA in latest season for prediction candidates (default: from config)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models on all data instead of loading saved models",
    )
    parser.add_argument(
        "--with-public",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include public projections comparison (default: True)",
    )
    parser.add_argument(
        "--fetch-public",
        action="store_true",
        help="Fetch public projections on-the-fly if not cached",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        data_config = yaml.safe_load(f)

    min_pa = args.min_pa or data_config.get("min_pa_predict", 300)
    end_year = data_config.get("end_year", 2025)
    predict_year = data_config.get("predict_year", 2026)
    seed = data_config.get("seed", 42)

    # Load merged data
    logger.info("Loading merged data from %s", args.data)
    df = pd.read_parquet(args.data)
    logger.info("Loaded %d rows, %d players", len(df), df["mlbam_id"].nunique())

    # Build features
    logger.info("Building features …")
    df = build_features(df, data_config)

    # Production split: retrain on all data with valid targets, predict end_year → predict_year
    retrain_data, predict_data = get_production_data(
        df, end_year=end_year, target_cols=TARGET_COLUMNS
    )

    # Filter prediction candidates by minimum PA
    predict_data = predict_data[predict_data["pa"] >= min_pa].copy()
    logger.info(
        "Retrain: %d rows | Predict: %d batters (PA >= %d in %d)",
        len(retrain_data),
        len(predict_data),
        min_pa,
        end_year,
    )

    # Split retrain data: hold out most recent year as eval_set for early stopping.
    # This prevents MTL from overfitting when training on all historical data.
    max_retrain_year = retrain_data["season"].max()
    train_part = retrain_data[retrain_data["season"] != max_retrain_year]
    eval_part = retrain_data[retrain_data["season"] == max_retrain_year]
    X_train, y_train = extract_xy(train_part, data_config)
    X_eval, y_eval = extract_xy(eval_part, data_config)
    eval_set: tuple[pd.DataFrame, pd.DataFrame] | None = (X_eval, y_eval)
    logger.info(
        "  Train: %d rows (≤%d) | Eval: %d rows (%d)",
        len(train_part), max_retrain_year - 1, len(eval_part), max_retrain_year,
    )
    X_predict, _ = extract_xy(predict_data, data_config)

    # Validation split for metrics reporting.
    val_year = end_year - 1
    val_data = df[(df["season"] == val_year) & df["target_obp"].notna()].copy()
    X_val: pd.DataFrame | None = None
    y_val: pd.DataFrame | None = None
    if len(val_data) > 0:
        X_val, y_val = extract_xy(val_data, data_config)

    # Load or train MTL model
    mtl_display = MODEL_CONFIGS["mtl"]["display_name"]
    train_season = train_part["season"].values if "season" in train_part.columns else None
    try:
        model = _load_or_train_model(
            X_train, y_train, args.retrain, seed,
            eval_set=eval_set, season=train_season,
        )
    except Exception as e:
        logger.error("Failed to load/train MTL model: %s", e)
        sys.exit(1)

    # Validation evaluation
    if X_val is not None and y_val is not None:
        val_preds = model.predict(align_features(X_val, model, mtl_display))
        val_metrics = compute_metrics(y_val, val_preds, TARGET_DISPLAY)

        prev_cols = [f"prev_year_{s}" for s in TARGET_STATS]
        val_baseline = None
        if all(c in val_data.columns for c in prev_cols):
            naive_vals = val_data[prev_cols].reset_index(drop=True)
            # When rate_targets is on, y_val contains per-PA rates for
            # count stats but naive_vals are raw counts. Convert naive
            # count stats to per-PA rates to match y_val's units.
            rate_targets_flag = data_config.get("rate_targets", False)
            if rate_targets_flag:
                # Use prev_year_pa (Y-1 PA) as denominator — this is the
                # correct PA for the season the naive counts come from.
                if "prev_year_pa" in val_data.columns:
                    denom = val_data["prev_year_pa"].reset_index(drop=True).astype(float)
                elif "pa" in val_data.columns:
                    denom = val_data["pa"].reset_index(drop=True).astype(float)
                else:
                    denom = pd.Series(1.0, index=naive_vals.index)
                for stat in COUNT_STATS:
                    col = f"prev_year_{stat}"
                    if col in naive_vals.columns:
                        naive_vals[col] = np.where(
                            denom > 0, naive_vals[col] / denom, np.nan,
                        )
            y_val_r = y_val.reset_index(drop=True)
            val_preds_r = val_preds.reset_index(drop=True)
            valid = naive_vals.notna().all(axis=1)
            if valid.sum() > 0:
                val_baseline = compute_naive_baseline(
                    y_val_r[valid], naive_vals[valid], TARGET_DISPLAY,
                )
                val_metrics = compute_metrics(
                    y_val_r[valid], val_preds_r[valid], TARGET_DISPLAY,
                )
        print_report(val_metrics, val_baseline, model_name=mtl_display,
                     split_name=f"validation ({val_year}→{end_year})")

    # Generate predictions
    predictions = model.predict(align_features(X_predict, model, mtl_display))

    # Convert rate predictions to counting stats if rate_targets is enabled
    rate_targets = data_config.get("rate_targets", False)
    projected_pa: pd.Series | None = None
    if rate_targets:
        season_games = data_config.get("season_games", {})
        projected_pa = project_pa(predict_data, season_games)
        logger.info(
            "PA projection (Marcel): mean=%.0f, median=%.0f, range=[%.0f, %.0f]",
            projected_pa.mean(), projected_pa.median(),
            projected_pa.min(), projected_pa.max(),
        )
        predictions = rate_to_count(predictions, projected_pa)

    # Build results table with player info (include idfg for projection join)
    results = pd.DataFrame({
        "_idfg": predict_data["idfg"].values,
        "Player": predict_data["name"].values,
        "Team": predict_data["team"].values,
        f"Age ({predict_year})": (predict_data["age"].values + 1).astype(int),
        f"{end_year} PA": predict_data["pa"].values.astype(int),
        f"{end_year} OBP": predict_data["obp"].values,
        f"{end_year} SLG": predict_data["slg"].values,
        f"{end_year} HR": predict_data["hr"].values.astype(int),
        f"{end_year} R": predict_data["r"].values.astype(int),
        f"{end_year} RBI": predict_data["rbi"].values.astype(int),
        f"{end_year} SB": predict_data["sb"].values.astype(int),
    })

    # Add projected PA column if rate targets are used
    if projected_pa is not None:
        results[f"{predict_year} PA (proj)"] = projected_pa.values.astype(int)

    # Add MTL prediction columns
    pred_cols = _build_single_model_results(predictions, mtl_display)
    for col_name, values in pred_cols.items():
        results[col_name] = values
    _round_and_clip(results, mtl_display)

    # Sort by predicted OPS
    results["Sort OPS"] = results[f"{mtl_display} OBP"] + results[f"{mtl_display} SLG"]
    results = results.sort_values("Sort OPS", ascending=False).reset_index(drop=True)
    results = results.drop(columns=["Sort OPS"])

    # --- Load and merge public projections ---
    proj_systems_present: list[str] = []
    if args.with_public:
        ext_dir = data_config.get("external_projections_dir", "data/external_projections")
        proj_config = data_config.get("projections", {})
        proj_systems = proj_config.get("systems", list(PROJECTION_SYSTEMS.keys()))

        if args.fetch_public:
            logger.info("Fetching external projections for %d …", predict_year)
            fetch_all_projections(predict_year, systems=proj_systems, out_dir=ext_dir)

        projections = _load_external_projections(predict_year, proj_systems, ext_dir)
        if projections is not None and len(projections) > 0:
            results = _merge_projections(results, projections)
            proj_systems_present = list(projections["projection_system"].unique())
            logger.info(
                "Merged %d projection system(s): %s",
                len(proj_systems_present),
                proj_systems_present,
            )
        else:
            logger.info("No cached projections found — run with --fetch-public to download")

    # Drop internal idfg column before display/CSV output.
    if "_idfg" in results.columns:
        results = results.drop(columns=["_idfg"])

    print(f"\n{'=' * 95}")
    print(f"  {predict_year} MLB Season Projections — {mtl_display}")
    print(f"  {len(results)} qualified batters (PA >= {min_pa})")
    print(f"{'=' * 95}")

    print(
        f"\n  {'#':>3} {'Player':<24} {'Team':<5} {'Age':>3}  "
        f"{'OBP':>6} {'SLG':>6} {'HR':>4} {'R':>4} {'RBI':>4} {'SB':>4}"
    )
    print(f"  {'-' * 85}")

    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(
            f"  {i:>3} {row['Player']:<24} {row['Team']:<5} "
            f"{row[f'Age ({predict_year})']:>3}  "
            f"{row[f'{mtl_display} OBP']:>6.3f} {row[f'{mtl_display} SLG']:>6.3f} "
            f"{row[f'{mtl_display} HR']:>4} {row[f'{mtl_display} R']:>4} "
            f"{row[f'{mtl_display} RBI']:>4} {row[f'{mtl_display} SB']:>4}"
        )

    print(f"\n{'=' * 95}")

    # Print projection comparison tables if available.
    if proj_systems_present:
        _print_projection_comparison(results, mtl_display, predict_year, proj_systems_present)

    # Save CSV files
    _PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # MTL projections CSV
    model_cols = ["Player", "Team", f"Age ({predict_year})", f"{end_year} PA"]
    for stat in TARGET_STATS:
        model_cols.append(f"{end_year} {stat.upper()}")
    for stat in TARGET_STATS:
        model_cols.append(f"{mtl_display} {stat.upper()}")
    mtl_path = _PROJECTIONS_DIR / f"projections_mtl_{predict_year}.csv"
    results[model_cols].to_csv(mtl_path, index=False)
    logger.info("Saved MTL projections → %s", mtl_path)

    # Comparison CSV with public projections
    if proj_systems_present:
        comp_cols = ["Player", "Team", f"Age ({predict_year})"]
        for stat in TARGET_STATS:
            upper = stat.upper()
            comp_cols.append(f"{mtl_display} {upper}")
            for s in proj_systems_present:
                pd_name = PROJ_DISPLAY_NAMES.get(s, s)
                col = f"{pd_name} {upper}"
                if col in results.columns:
                    comp_cols.append(col)
        available_comp = [c for c in comp_cols if c in results.columns]
        comp_path = _PROJECTIONS_DIR / f"projections_vs_external_{predict_year}.csv"
        results[available_comp].to_csv(comp_path, index=False)
        logger.info("Saved projection comparison → %s", comp_path)

    # MTL task weights
    if hasattr(model, "get_learned_task_weights"):
        weights = model.get_learned_task_weights()
        print("  MTL — Learned task weights (higher = more confident):")
        for target, w in weights.items():
            label = target.replace("target_", "").upper()
            print(f"    {label:<6} {w:.4f}")
        print()


if __name__ == "__main__":
    main()

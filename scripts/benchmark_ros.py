"""Rolling ROS (rest-of-season) benchmark for the in-season projection pipeline.

Evaluates baselines at per-player PA checkpoints (50 / 100 / 200 / 400) across
2023-2025 weekly snapshots. Baselines (all emit rate predictions for the
six ROS targets: OBP, SLG, HR/PA, R/PA, RBI/PA, SB/PA):

* **persist_observed** — predict ROS rate = observed ytd rate. No model needed.
* **frozen_preseason** — predict ROS rate = preseason MTL rate. Requires a
  preseason predictions parquet per year (either cached or regenerated).
* **marcel_blend** — PA-weighted blend of ytd and preseason rates with a
  prior-weight ``prior_pa`` (default 200). Requires preseason predictions.
* **shrinkage** — Beta-Binomial posterior mean combining the preseason prior
  with observed ytd counts. Also emits Beta-CDF quantiles for pinball/PIT.
* **phase2** — quantile-regressing MTL trained on weekly snapshots. Expensive;
  retrains an ensemble per eval year from all earlier snapshots. Emits full
  quantile arrays so pinball loss / PIT coverage are computed alongside RMSE.

Usage
-----
    uv run python scripts/benchmark_ros.py --years 2023 2024 2025
    uv run python scripts/benchmark_ros.py --years 2024 --include persist_observed

    # Retrain preseason MTL for each year and cache (expensive):
    uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --retrain

    # Phase 2 ensemble + quantile metrics (requires weekly snapshot data):
    uv run python scripts/benchmark_ros.py --years 2023 2024 2025 --include phase2 --retrain
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.splits import get_production_data
from src.eval.metrics import compute_metrics
from src.eval.plots import plot_pit_histogram
from src.eval.ros_metrics import (
    DEFAULT_PA_CHECKPOINTS,
    DEFAULT_QUANTILE_LEVELS,
    ROS_RATE_TARGETS,
    ROS_TARGET_DISPLAY,
    ROS_TARGET_STATS,
    ROS_YTD_RATES,
    pa_checkpoint_rows,
    pit_coverage,
    quantile_loss,
)
from src.features.pipeline import build_features, extract_xy
from src.features.registry import PLAYER_SEASON_KEY, TARGET_COLUMNS
from src.models.baselines.shrinkage import (
    DEFAULT_QUANTILE_TAUS,
    fit_tau_per_stat,
    predict_shrinkage,
    predict_shrinkage_quantiles,
)
from src.models.utils import align_features, get_model_configs, train_model_for_year

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = get_model_configs()

_BASELINES_NO_PRESEASON = ("persist_observed",)
# Preseason-dependent baselines — all need the per-year preseason cache frame
# for alignment. phase2 additionally needs the weekly snapshot files for its
# own training pass.
_BASELINES_NEED_PRESEASON = ("frozen_preseason", "marcel_blend", "shrinkage", "phase2")
ALL_BASELINES = (*_BASELINES_NO_PRESEASON, *_BASELINES_NEED_PRESEASON)

# Baselines that emit full quantile arrays (shape (n, 6, n_taus)) alongside
# their point estimate. These are the ones that participate in pinball / PIT
# metrics; point-only baselines contribute to RMSE only.
_BASELINES_EMIT_QUANTILES = ("shrinkage", "phase2")

_BASELINE_DISPLAY = {
    "persist_observed": "PersistObs",
    "frozen_preseason": "FrozenPre",
    "marcel_blend": "MarcelBlend",
    "shrinkage": "Shrinkage",
    "phase2": "Phase2",
}

_DEFAULT_CACHE_DIR = Path("data/reports/benchmark_ros/preseason")
_DEFAULT_OUTPUT_DIR = Path("data/reports/benchmark_ros")
_DEFAULT_PHASE2_CACHE_DIR = Path("data/reports/benchmark_ros/phase2")
_DEFAULT_PHASE2_CONFIG = Path("configs/mtl_ros.yaml")


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
        df_featured,
        end_year=eval_year - 1,
        target_cols=TARGET_COLUMNS,
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
    baselines). With ``retrain=True`` the cache is regenerated even when a
    stale file already exists, so feature/model/config changes are reflected.
    """
    cache_path = cache_dir / f"mtl_preseason_{year}.parquet"
    if cache_path.exists() and not retrain:
        logger.info("  Loaded preseason cache %s", cache_path)
        return pd.read_parquet(cache_path)
    if cache_path.exists() and retrain:
        logger.info("  Overwriting preseason cache at %s (--retrain)", cache_path)

    if not retrain:
        logger.warning(
            "  Missing preseason cache for %d (%s); skipping preseason-dependent "
            "baselines. Pass --retrain to regenerate.",
            year,
            cache_path,
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
# Phase 2 (quantile MTL) training + scoring
# ---------------------------------------------------------------------------


def _load_phase2_config(
    config_path: Path,
    epochs_override: int | None,
    n_seeds_override: int | None,
) -> dict:
    """Load the Phase 2 YAML config with optional benchmark-speed overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("training", {})
    cfg.setdefault("ensemble", {})
    if epochs_override is not None:
        cfg["training"]["epochs"] = int(epochs_override)
    if n_seeds_override is not None:
        cfg["ensemble"]["n_seeds"] = int(n_seeds_override)
    return cfg


def _phase2_training_snapshots_for_year(
    eval_year: int,
    raw_dir: Path,
) -> pd.DataFrame:
    """Load weekly snapshots for all seasons strictly before ``eval_year``.

    Raises ``FileNotFoundError`` with an actionable message if the snapshot
    directory is empty; callers handle that and skip the phase2 baseline.
    """
    candidate_paths = sorted(raw_dir.glob("weekly_snapshots_*.parquet"))
    frames: list[pd.DataFrame] = []
    for path in candidate_paths:
        # Filename is ``weekly_snapshots_{year}.parquet``.
        try:
            year = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if year < eval_year:
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"phase2 baseline requires weekly snapshots at "
            f"{raw_dir}/weekly_snapshots_{{year}}.parquet for years < {eval_year}; "
            f"run `uv run python -m src.data.build_snapshots --seasons <year>`"
        )
    return pd.concat(frames, ignore_index=True)


def _load_or_train_phase2_ensemble(
    eval_year: int,
    phase2_cache_dir: Path,
    phase2_config: dict,
    raw_dir: Path,
    df_featured: pd.DataFrame | None,
    retrain: bool,
):
    """Return a fitted ``MTLQuantileEnsembleForecaster`` for ``eval_year``.

    Mirrors the frozen_preseason cache pattern: train once and save under
    ``phase2_cache_dir/year_{eval_year}/``, then reload for subsequent runs.
    The training data covers every snapshot year strictly before ``eval_year``
    (train_end_season = eval_year - 1). Returns ``None`` on cache miss when
    ``retrain`` is False — callers then skip phase2 for this year rather
    than silently kicking off an expensive training run.
    """
    # Local imports: keep the benchmark importable even when torch is absent
    # (e.g. tests that only exercise point baselines).
    from src.models.mtl_ros.model import MTLQuantileEnsembleForecaster
    from src.models.mtl_ros.train import train_ros

    year_dir = phase2_cache_dir / f"year_{eval_year}"
    meta_path = year_dir / "ensemble_meta.json"
    if meta_path.exists() and not retrain:
        logger.info("  Loaded Phase 2 ensemble cache %s", year_dir)
        return MTLQuantileEnsembleForecaster.load(year_dir)

    if not retrain:
        logger.warning(
            "  Phase 2 cache missing for %d (%s); skipping phase2 for this year. "
            "Pass --retrain to train from scratch.",
            eval_year,
            year_dir,
        )
        return None

    snapshots = _phase2_training_snapshots_for_year(eval_year, raw_dir)
    preseason_df: pd.DataFrame | None = None
    if df_featured is not None:
        preseason_df = df_featured.loc[df_featured["season"] < eval_year]

    # Force splits to treat everything as training — we don't run an internal
    # val carve-out during the benchmark (saves an expensive epoch loop).
    phase2_config.setdefault("splits", {})
    phase2_config["splits"] = {
        "train_end_season": int(eval_year - 1),
    }
    logger.info(
        "  Training Phase 2 ensemble for eval_year=%d on %d snapshot rows "
        "(train_end_season=%d)",
        eval_year,
        len(snapshots),
        eval_year - 1,
    )
    ensemble = train_ros(
        phase2_config,
        snapshots_df=snapshots,
        preseason_df=preseason_df,
    )

    year_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(year_dir)
    logger.info("  Cached Phase 2 ensemble → %s", year_dir)
    return ensemble


def _phase2_eval_preseason_frame(
    eval_year: int,
    df_featured: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Return the preseason feature frame for ``eval_year`` scoring.

    The phase2 ensemble was trained with preseason features joined on
    ``(mlbam_id, season)`` from ``df_featured``. At eval time we need the
    SAME feature columns in the SAME raw space — just for the eval year's
    rows — so the predict-time reindex/standard-scale step stays in-distribution.
    Using the MTL preseason prediction cache (which only has ``target_*``
    columns) would leave every non-target feature missing and trigger the
    fillna-0 path, making phase2 metrics meaningless.
    """
    if df_featured is None:
        return None
    eval_pre = df_featured.loc[df_featured["season"] == eval_year]
    if eval_pre.empty:
        return None
    return eval_pre.copy()


def _phase2_feature_matrix(
    rows: pd.DataFrame,
    ensemble,
    phase2_config: dict,
    preseason: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Build the feature matrix the Phase 2 ensemble expects.

    The ensemble was trained on in-season features + a configured slice of
    preseason groups; ``forecaster.feature_names_`` captures the exact column
    set and order. We rebuild that matrix for the checkpoint rows with:

    * ``compute_in_season_features`` on the snapshot rows.
    * Left-join against the preseason frame (on ``mlbam_id``/``season``)
      to pick up the requested preseason groups.
    * Reindex to the trained feature order, filling missing columns with the
      training-time feature mean (so they map to 0 in scaled space — neutral
      rather than an extreme ``-mean/std`` value).

    Returns ``None`` when no feature names are recoverable from the ensemble
    (shouldn't happen for a fitted ensemble; defensive guard).
    """
    from src.features.in_season import compute_in_season_features

    feature_names: list[str] = list(ensemble.feature_names_)
    if not feature_names:
        return None

    in_season = compute_in_season_features(rows)
    enriched = pd.concat(
        [
            rows.drop(
                columns=[c for c in in_season.columns if c in rows.columns],
                errors="ignore",
            ),
            in_season,
        ],
        axis=1,
    )

    if preseason is not None:
        overlap = (set(enriched.columns) & set(preseason.columns)) - set(
            PLAYER_SEASON_KEY
        )
        pre = preseason.drop(columns=list(overlap)) if overlap else preseason
        enriched = enriched.merge(
            pre,
            on=list(PLAYER_SEASON_KEY),
            how="left",
        )

    # Reindex to the ensemble's trained feature order. Missing columns are
    # filled with the training-time feature mean; after standardization that
    # maps to exactly 0 (neutral), whereas filling with 0 in raw space would
    # map to ``-mean/std`` — extreme for features with non-zero mean (age ≈
    # -30/σ becomes a huge negative). Pulling ``scaler.mean_`` from the
    # first seed is safe — every seed shares an identical scaler.
    missing = [name for name in feature_names if name not in enriched.columns]
    if missing:
        sample = sorted(missing)[:10]
        logger.warning(
            "phase2 predict: %d features missing; filling with training means. "
            "Samples: %s%s",
            len(missing),
            sample,
            "..." if len(missing) > len(sample) else "",
        )

    X = enriched.reindex(columns=feature_names).astype(float)
    _fill_with_train_mean(X, ensemble, feature_names)
    return X


def _fill_with_train_mean(
    X: pd.DataFrame,
    ensemble,
    feature_names: list[str],
) -> None:
    """In-place NaN-fill using the ensemble's training-time feature means.

    Any feature whose training mean cannot be recovered (defensively: no
    fitted scaler on the first seed) falls back to 0.0 so the legacy
    behavior is preserved for unfitted or partially-configured ensembles.
    """
    # Ensemble stores forecasters_ list; pull feature_scaler_ from the first.
    forecasters = getattr(ensemble, "forecasters_", None)
    if not forecasters:
        X.fillna(0.0, inplace=True)
        return
    scaler = getattr(forecasters[0], "feature_scaler_", None)
    train_feature_names = list(getattr(forecasters[0], "feature_names_", []) or [])
    if scaler is None or not train_feature_names:
        X.fillna(0.0, inplace=True)
        return
    mean_by_name: dict[str, float] = dict(zip(train_feature_names, scaler.mean_))
    for col in feature_names:
        # Default to 0.0 for any feature not seen by the scaler (shouldn't
        # happen if feature_names matches the ensemble's trained names).
        fill_val = float(mean_by_name.get(col, 0.0))
        X[col] = X[col].fillna(fill_val)


def predict_phase2(
    rows: pd.DataFrame,
    ensemble,
    phase2_config: dict,
    preseason: pd.DataFrame | None,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Return (point_df, quantiles_array) for the Phase 2 baseline at the checkpoint.

    ``point_df`` is indexed by ``ROS_RATE_TARGETS`` and holds the median
    (tau=0.50) for each (row, target) — the apples-to-apples RMSE
    representative. ``quantiles_array`` is the full ``(n_rows, 6, n_taus)``
    tensor used for pinball loss + PIT. The ensemble already enforces
    tau-monotonicity inside ``predict``, so no extra sort is needed here.
    """
    X = _phase2_feature_matrix(rows, ensemble, phase2_config, preseason)
    if X is None or len(X) == 0:
        return None

    preds = ensemble.predict(X)
    q_arr: np.ndarray = np.asarray(preds["quantiles"], dtype=np.float64)

    # Median slice — prefer the exact 0.5 index when present, fall back to the
    # middle quantile (matches the network's ``median_index`` convention).
    taus = list(phase2_config.get("model", {}).get("taus", DEFAULT_QUANTILE_TAUS))
    try:
        median_idx = taus.index(0.5)
    except ValueError:
        median_idx = q_arr.shape[-1] // 2

    median = q_arr[:, :, median_idx]
    point_df = pd.DataFrame(median, columns=list(ROS_RATE_TARGETS))
    point_df.index = rows.index
    return point_df, q_arr


# ---------------------------------------------------------------------------
# Baseline predictions
# ---------------------------------------------------------------------------


def _ytd_rate_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-target ytd rate DataFrame aligned with ``ROS_RATE_TARGETS``."""
    missing = [c for c in ROS_YTD_RATES if c not in rows.columns]
    if missing:
        raise KeyError(f"Snapshot rows missing ytd rate columns: {missing}")
    out = pd.DataFrame(
        {tgt: rows[ytd].values for tgt, ytd in zip(ROS_RATE_TARGETS, ROS_YTD_RATES)}
    )
    return out


def _preseason_rate_matrix(
    rows: pd.DataFrame,
    preseason: pd.DataFrame,
    id_col: str = "mlbam_id",
    season_col: str = "season",
) -> pd.DataFrame | None:
    """Broadcast preseason rate predictions onto the checkpoint row order.

    The join is keyed on ``(id_col, season_col)`` when both frames carry
    ``season`` so multi-season rows don't collapse onto the same player's
    first-season prior. When either side lacks ``season`` we fall back to the
    id-only join (safe for single-year calls).

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

    use_season = season_col in rows.columns and season_col in preseason.columns
    if use_season:
        key_cols = [id_col, season_col]
        pre_indexed = preseason.drop_duplicates(subset=key_cols).set_index(key_cols)[
            pre_target_cols
        ]
        keys = pd.MultiIndex.from_arrays(
            [rows[id_col].values, rows[season_col].values],
            names=key_cols,
        )
        aligned = pre_indexed.reindex(keys).reset_index(drop=True)
    else:
        pre_indexed = preseason.drop_duplicates(subset=[id_col]).set_index(id_col)[
            pre_target_cols
        ]
        aligned = pre_indexed.reindex(rows[id_col].values).reset_index(drop=True)
    aligned.columns = list(ROS_RATE_TARGETS)
    # Zero-overlap cache: an all-NaN matrix would drive the downstream
    # non-NaN intersection filter to empty, wiping even baselines that don't
    # need preseason (e.g. persist_observed). Treat as if preseason were absent.
    if aligned.isna().all(axis=None):
        logger.warning(
            "Preseason cache has zero %s overlap with checkpoint rows (%d rows); "
            "skipping preseason-dependent baselines.",
            id_col,
            len(rows),
        )
        return None
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


def _stack_quantile_dict(
    per_target: dict[str, pd.DataFrame],
    taus: Sequence[float],
) -> np.ndarray:
    """Convert ``{ros_target: DataFrame[q columns]}`` → ``(n, T, Q)`` ndarray.

    Assumes the per-target DataFrames share a row index and the column order
    matches ``taus``. ``ROS_RATE_TARGETS`` defines the target (second) axis.
    """
    q_cols = [f"q{tau:.2f}".replace(".", "_") for tau in taus]
    stacks = []
    for tgt in ROS_RATE_TARGETS:
        df = per_target[tgt]
        stacks.append(df.reindex(columns=q_cols).to_numpy(dtype=np.float64))
    # shape: (T, n, Q) → transpose to (n, T, Q)
    return np.stack(stacks, axis=0).transpose(1, 0, 2)


def evaluate_checkpoint(
    checkpoint_rows: pd.DataFrame,
    baselines: list[str],
    preseason: pd.DataFrame | None,
    prior_pa: float,
    min_ros_pa: int,
    shrinkage_tau: dict[str, float] | None = None,
    phase2_ensemble=None,
    phase2_config: dict | None = None,
    phase2_preseason: pd.DataFrame | None = None,
    quantile_taus: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> dict:
    """Compute point + quantile metrics for each baseline at one PA checkpoint.

    Apples-to-apples RMSE is reported for every active baseline using its
    point estimate (median quantile for shrinkage/phase2). Baselines that
    emit full quantile arrays additionally contribute pinball loss + PIT
    coverage via the ``quantiles`` and ``quantile_metrics`` keys.

    Returns ``{"n_players": int, "y_true": np.ndarray, "systems": {baseline: {...}}}``,
    or ``{"n_players": 0, "systems": {}}`` when no baselines run or no rows
    survive filtering. Per-system payload keys:

    * ``"y_pred"`` — ``(n, 6)`` point estimates (median for quantile baselines).
    * ``"metrics"`` — :func:`compute_metrics` output for point estimates.
    * ``"quantiles"`` (optional) — ``(n, 6, n_taus)`` quantile predictions.
    * ``"quantile_metrics"`` (optional) — pinball + PIT dict.
    """
    # Per-baseline predictions. Build the preseason→rows matrix at most once:
    # both frozen_preseason and marcel_blend reuse the same alignment.
    predictions: dict[str, pd.DataFrame] = {}
    # Optional per-baseline quantile arrays (same row order as predictions).
    quantile_preds: dict[str, np.ndarray] = {}

    if "persist_observed" in baselines:
        predictions["persist_observed"] = predict_persist_observed(checkpoint_rows)

    frozen_matrix: pd.DataFrame | None = None
    if preseason is not None and any(b in _BASELINES_NEED_PRESEASON for b in baselines):
        frozen_matrix = _preseason_rate_matrix(checkpoint_rows, preseason)

    if "frozen_preseason" in baselines and frozen_matrix is not None:
        predictions["frozen_preseason"] = frozen_matrix
    if "marcel_blend" in baselines and frozen_matrix is not None:
        blended = predict_marcel_blend(
            checkpoint_rows,
            preseason,
            prior_pa=prior_pa,
            preseason_matrix=frozen_matrix,
        )
        if blended is not None:
            predictions["marcel_blend"] = blended
    if "shrinkage" in baselines and frozen_matrix is not None:
        shrunk = predict_shrinkage(
            checkpoint_rows,
            tau_per_stat=shrinkage_tau,
            preseason_matrix=frozen_matrix,
        )
        if shrunk is not None:
            predictions["shrinkage"] = shrunk
            # Also emit Beta-CDF quantiles for the same tau grid so pinball
            # loss / PIT can be computed alongside the RMSE.
            shrunk_quantiles = predict_shrinkage_quantiles(
                checkpoint_rows,
                tau_per_stat=shrinkage_tau,
                preseason_matrix=frozen_matrix,
                taus=quantile_taus,
            )
            if shrunk_quantiles is not None:
                quantile_preds["shrinkage"] = _stack_quantile_dict(
                    shrunk_quantiles,
                    quantile_taus,
                )

    if "phase2" in baselines and phase2_ensemble is not None:
        # Phase 2 must score against the same feature frame it trained with —
        # the preseason MTL prediction cache (which only has target_* columns)
        # would leave age/park/temporal missing and fill them with zero after
        # scaling. Fall back to ``preseason`` only when no phase2-specific
        # frame is available (e.g. when --retrain was not passed and
        # df_featured wasn't loaded).
        phase2_pre_arg = phase2_preseason if phase2_preseason is not None else preseason
        phase2_preds = predict_phase2(
            checkpoint_rows,
            phase2_ensemble,
            phase2_config or {},
            phase2_pre_arg,
        )
        if phase2_preds is not None:
            point_df, q_arr = phase2_preds
            predictions["phase2"] = point_df
            quantile_preds["phase2"] = q_arr

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

    valid_mask = valid.to_numpy(dtype=bool)
    y_true_f = y_true.loc[valid].reset_index(drop=True)
    systems: dict[str, dict] = {}
    for name, pred in predictions.items():
        pred_f = pred.loc[valid].reset_index(drop=True)
        payload: dict = {
            "y_pred": pred_f.values,
            "metrics": compute_metrics(
                y_true_f,
                pred_f,
                list(ROS_TARGET_DISPLAY),
            ),
        }
        # Quantile metrics: only baselines that produced a full quantile
        # array participate here. Point-only baselines stay out of the
        # pinball/PIT tables so the comparison is apples-to-apples per column.
        if name in quantile_preds:
            q_full = quantile_preds[name][valid_mask]
            payload["quantiles"] = q_full
            payload["quantile_metrics"] = {
                "pinball": quantile_loss(
                    y_true_f,
                    q_full,
                    list(quantile_taus),
                    target_names=list(ROS_TARGET_DISPLAY),
                ),
                "pit": pit_coverage(
                    y_true_f,
                    q_full,
                    list(quantile_taus),
                    target_names=list(ROS_TARGET_DISPLAY),
                ),
            }
        systems[name] = payload

    return {"n_players": n, "y_true": y_true_f.values, "systems": systems}


def evaluate_year(
    year: int,
    snapshots: pd.DataFrame,
    thresholds: list[int],
    baselines: list[str],
    preseason: pd.DataFrame | None,
    prior_pa: float,
    min_ros_pa: int,
    shrinkage_tau: dict[str, float] | None = None,
    phase2_ensemble=None,
    phase2_config: dict | None = None,
    phase2_preseason: pd.DataFrame | None = None,
    quantile_taus: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> dict:
    """Run all thresholds for one season's snapshots."""
    yearly = snapshots[snapshots["season"] == year]
    checkpoints = pa_checkpoint_rows(yearly, thresholds=thresholds)
    out: dict = {"year": year, "thresholds": {}}
    for t, rows in checkpoints.items():
        result = evaluate_checkpoint(
            rows,
            baselines,
            preseason,
            prior_pa,
            min_ros_pa,
            shrinkage_tau=shrinkage_tau,
            phase2_ensemble=phase2_ensemble,
            phase2_config=phase2_config,
            phase2_preseason=phase2_preseason,
            quantile_taus=quantile_taus,
        )
        out["thresholds"][t] = result
        sys_summary = (
            ", ".join(
                f"{_BASELINE_DISPLAY.get(k, k)}={v['metrics']['aggregate']['rmse']:.4f}"
                for k, v in result["systems"].items()
            )
            or "(none)"
        )
        logger.info(
            "  %d @ %d PA: n=%d | %s",
            year,
            t,
            result["n_players"],
            sys_summary,
        )
    return out


# ---------------------------------------------------------------------------
# Pooling & reporting
# ---------------------------------------------------------------------------


def pool_by_threshold(
    year_results: list[dict],
    quantile_taus: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> dict:
    """Concatenate y_true/y_pred across years at each threshold.

    Returns ``{threshold: {"systems": {name: metrics}, "n_players_per_system": ...,
    "quantile_metrics_per_system": {...}}}``. Quantile metrics are included
    only for systems that produced full quantile arrays at every pooled year.
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
        quantile_metrics_per_system: dict[str, dict] = {}
        pooled_quantile_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name in system_names:
            y_trues, y_preds, y_quantiles = [], [], []
            for yr in year_results:
                cell = yr["thresholds"].get(t, {})
                if name not in cell.get("systems", {}):
                    continue
                y_trues.append(cell["y_true"])
                sys_payload = cell["systems"][name]
                y_preds.append(sys_payload["y_pred"])
                if "quantiles" in sys_payload:
                    y_quantiles.append(sys_payload["quantiles"])
            if not y_trues:
                continue
            yt = np.concatenate(y_trues, axis=0)
            yp = np.concatenate(y_preds, axis=0)
            per_system[name] = compute_metrics(
                yt,
                yp,
                list(ROS_TARGET_DISPLAY),
            )
            n_players_accum[name] = int(yt.shape[0])

            # Quantile metrics pool iff every contributing year had
            # quantile arrays for this system. Otherwise we'd bias the
            # per-system pinball with whichever years happened to have them.
            if y_quantiles and len(y_quantiles) == len(y_trues):
                yq = np.concatenate(y_quantiles, axis=0)
                quantile_metrics_per_system[name] = {
                    "pinball": quantile_loss(
                        yt,
                        yq,
                        list(quantile_taus),
                        target_names=list(ROS_TARGET_DISPLAY),
                    ),
                    "pit": pit_coverage(
                        yt,
                        yq,
                        list(quantile_taus),
                        target_names=list(ROS_TARGET_DISPLAY),
                    ),
                }
                pooled_quantile_inputs[name] = (yt, yq)

        pooled[t] = {
            "systems": per_system,
            "n_players_per_system": n_players_accum,
            "quantile_metrics_per_system": quantile_metrics_per_system,
            "_quantile_inputs": pooled_quantile_inputs,
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
        print(
            f"\n  [{t} PA checkpoint]  n per system: "
            + ", ".join(f"{_BASELINE_DISPLAY.get(s, s)}={n_by_sys[s]}" for s in systems)
        )
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

        # Quantile metrics section: only systems that emitted full quantile
        # arrays participate — the rest stay out so the table stays honest.
        qmetrics = cell.get("quantile_metrics_per_system", {})
        if qmetrics:
            print(f"\n  [{t} PA checkpoint]  Pinball loss (lower is better):")
            print(f"  {'System':<14}", end="")
            for tgt in ROS_TARGET_DISPLAY:
                print(f"  {tgt:>8}", end="")
            print(f"  {'Mean':>8}")
            print(f"  {'-' * (14 + 10 * (len(ROS_TARGET_DISPLAY) + 1))}")
            for name in system_order:
                if name not in qmetrics:
                    continue
                pb = qmetrics[name]["pinball"]
                print(f"  {_BASELINE_DISPLAY.get(name, name):<14}", end="")
                for tgt in ROS_TARGET_DISPLAY:
                    print(f"  {pb['per_target'][tgt]['mean_pinball']:>8.4f}", end="")
                print(f"  {pb['aggregate']['mean_pinball']:>8.4f}")

            print(
                f"\n  [{t} PA checkpoint]  PIT coverage (nominal vs empirical aggregate):"
            )
            for name in system_order:
                if name not in qmetrics:
                    continue
                agg = qmetrics[name]["pit"]["aggregate"]
                nominal_strs = ", ".join(
                    f"{lv:.2f}→{agg[lv]:.3f}" for lv in sorted(agg)
                )
                print(f"    {_BASELINE_DISPLAY.get(name, name):<14}  {nominal_strs}")
    print()


def _save_pit_plots(
    pooled: dict[int, dict],
    output_dir: Path,
    quantile_taus: Sequence[float] = DEFAULT_QUANTILE_LEVELS,
) -> None:
    """Write per-baseline PIT histograms pooled across thresholds.

    Only baselines that produced full quantile arrays at every threshold (the
    ``_quantile_inputs`` stash from :func:`pool_by_threshold`) get a plot;
    the rest silently skip.
    """
    # Aggregate y_true / y_quantiles per baseline across all thresholds to
    # maximise the PIT sample size — the per-threshold n is fairly small.
    per_baseline: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for cell in pooled.values():
        for name, (yt, yq) in cell.get("_quantile_inputs", {}).items():
            per_baseline.setdefault(name, ([], []))
            per_baseline[name][0].append(yt)
            per_baseline[name][1].append(yq)

    for name, (yts, yqs) in per_baseline.items():
        if not yts:
            continue
        y_true = np.concatenate(yts, axis=0)
        y_quant = np.concatenate(yqs, axis=0)
        plot_path = output_dir / f"pit_{name}.png"
        plot_pit_histogram(
            y_true,
            y_quant,
            list(quantile_taus),
            target_names=list(ROS_TARGET_DISPLAY),
            path=plot_path,
        )
        logger.info("Saved PIT plot → %s", plot_path)


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
    pit_plot: bool = False,
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
                name: cell["systems"][name]
                for name in system_order
                if name in cell["systems"]
            },
            "quantile_metrics": {
                name: cell["quantile_metrics_per_system"][name]
                for name in system_order
                if name in cell.get("quantile_metrics_per_system", {})
            },
        }
    for yr in year_results:
        yr_entry: dict = {"year": yr["year"], "thresholds": {}}
        for t, cell in yr["thresholds"].items():
            per_year_systems: dict = {}
            for name in cell.get("systems", {}):
                sys_payload = cell["systems"][name]
                entry: dict = {"metrics": sys_payload["metrics"]}
                if "quantile_metrics" in sys_payload:
                    entry["quantile_metrics"] = sys_payload["quantile_metrics"]
                per_year_systems[name] = entry
            yr_entry["thresholds"][str(t)] = {
                "n_players": cell.get("n_players", 0),
                "systems": per_year_systems,
            }
        json_report["per_year"].append(yr_entry)

    json_path = output_dir / "benchmark_ros_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    logger.info("Saved JSON report → %s", json_path)

    # Optional PIT plots per baseline that emitted quantile arrays. We pool
    # rows across thresholds so each plot has the full n for its system.
    if pit_plot:
        _save_pit_plots(pooled, output_dir)

    rows: list[dict] = []
    for t in sorted(pooled):
        cell = pooled[t]
        for name in system_order:
            metrics = cell["systems"].get(name)
            if metrics is None:
                continue
            rows.append(
                {
                    "threshold": t,
                    "system": name,
                    "n_players": cell["n_players_per_system"].get(name, 0),
                    "target": "Mean",
                    "rmse": metrics["aggregate"]["rmse"],
                    "mae": metrics["aggregate"]["mae"],
                    "r2": metrics["aggregate"]["r2"],
                    "mape": metrics["aggregate"]["mape"],
                }
            )
            for tgt in ROS_TARGET_DISPLAY:
                pt = metrics["per_target"][tgt]
                rows.append(
                    {
                        "threshold": t,
                        "system": name,
                        "n_players": cell["n_players_per_system"].get(name, 0),
                        "target": tgt,
                        "rmse": pt["rmse"],
                        "mae": pt["mae"],
                        "r2": pt["r2"],
                        "mape": pt["mape"],
                    }
                )
    csv_path = output_dir / "benchmark_ros_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info("Saved CSV table → %s", csv_path)


# ---------------------------------------------------------------------------
# Shrinkage tau fitting
# ---------------------------------------------------------------------------


def _fit_shrinkage_tau_from_years(
    fit_years: list[int],
    snapshots: pd.DataFrame,
    thresholds: list[int],
    cache_dir: Path,
) -> dict[str, float] | None:
    """Fit per-stat pseudocounts on the given ``fit_years`` only.

    Collects PA-checkpoint rows and preseason caches strictly from
    ``fit_years`` and runs ``fit_tau_per_stat`` on the concatenated frame.
    Returns ``None`` if no caches exist for any ``fit_years``; callers should
    then fall back to ``DEFAULT_TAU0``.

    Callers are responsible for ensuring ``fit_years`` is disjoint from the
    years being scored (e.g. leave-one-year-out) so the reported metrics
    aren't optimistically biased by the fit.
    """
    if not fit_years:
        return None

    training_frames: list[pd.DataFrame] = []
    preseason_frames: list[pd.DataFrame] = []
    for year in fit_years:
        cache_path = cache_dir / f"mtl_preseason_{year}.parquet"
        if not cache_path.exists():
            continue
        year_snapshots = snapshots[snapshots["season"] == year]
        checkpoints = pa_checkpoint_rows(year_snapshots, thresholds=thresholds)
        for rows in checkpoints.values():
            if len(rows):
                training_frames.append(rows)
        preseason_frames.append(pd.read_parquet(cache_path))

    if not training_frames or not preseason_frames:
        return None

    training_rows = pd.concat(training_frames, ignore_index=True)
    preseason = pd.concat(preseason_frames, ignore_index=True)
    return fit_tau_per_stat(training_rows, preseason)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROS benchmark: evaluate weekly baselines at PA checkpoints.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2023, 2024, 2025],
        help="Evaluation years (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_PA_CHECKPOINTS),
        help=f"PA checkpoints to evaluate (default: {list(DEFAULT_PA_CHECKPOINTS)})",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=list(ALL_BASELINES),
        default=list(ALL_BASELINES),
        help=f"Baselines to include (default: all {list(ALL_BASELINES)})",
    )
    parser.add_argument(
        "--prior-pa",
        type=float,
        default=200.0,
        help="Prior PA weight for marcel_blend (default: 200)",
    )
    parser.add_argument(
        "--min-ros-pa",
        type=int,
        default=50,
        help="Minimum ros_pa required for a row to count (default: 50)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing weekly_snapshots_{year}.parquet files",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_DEFAULT_CACHE_DIR),
        help="Directory for per-year preseason MTL prediction cache",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Output directory for report JSON + CSV",
    )
    parser.add_argument(
        "--data",
        default="data/merged_batter_data.parquet",
        help="Merged dataset for preseason retraining (only used with --retrain)",
    )
    parser.add_argument(
        "--data-config",
        default="configs/data.yaml",
        help="Data config YAML (only used with --retrain)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain preseason MTL for each year when cache is missing (slow).",
    )
    parser.add_argument(
        "--fit-shrinkage-tau",
        action="store_true",
        help="Fit shrinkage pseudocounts per stat via leave-one-year-out "
        "cross-fitting across --years (needs >= 2 eval years). Default "
        "uses DEFAULT_TAU0.",
    )
    parser.add_argument(
        "--phase2-config",
        type=str,
        default=str(_DEFAULT_PHASE2_CONFIG),
        help="YAML config for the Phase 2 ensemble (used when phase2 is in --include).",
    )
    parser.add_argument(
        "--phase2-epochs",
        type=int,
        default=None,
        help="Override training.epochs in the Phase 2 config for benchmark speed.",
    )
    parser.add_argument(
        "--phase2-n-seeds",
        type=int,
        default=None,
        help="Override ensemble.n_seeds in the Phase 2 config.",
    )
    parser.add_argument(
        "--phase2-cache-dir",
        type=str,
        default=str(_DEFAULT_PHASE2_CACHE_DIR),
        help="Directory for per-year Phase 2 ensemble checkpoints.",
    )
    parser.add_argument(
        "--pit-plot",
        action="store_true",
        help="Emit pit_{baseline}.png plots for every baseline that produced "
        "full quantile arrays (shrinkage, phase2).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    phase2_cache_dir = Path(args.phase2_cache_dir)
    years = sorted(args.years)

    logger.info(
        "Loading weekly snapshots for %s from %s …",
        years,
        args.raw_dir,
    )
    snapshots = load_weekly_snapshots(years, raw_dir=args.raw_dir)

    df_featured: pd.DataFrame | None = None
    data_config: dict | None = None
    # phase2 needs preseason features for the join, same as the other
    # preseason-dependent baselines; load the merged frame once when requested.
    need_preseason = any(b in args.include for b in _BASELINES_NEED_PRESEASON)
    want_phase2 = "phase2" in args.include
    if (need_preseason and args.retrain) or want_phase2:
        logger.info("Loading merged data + building features (for retraining) …")
        df = pd.read_parquet(args.data)
        with open(args.data_config) as f:
            data_config = yaml.safe_load(f)
        df_featured = build_features(df, data_config)

    # Warm the preseason cache before fitting tau, otherwise a --retrain run
    # with an empty cache directory would fit on no data and silently fall
    # back to DEFAULT_TAU0.
    preseason_by_year: dict[int, pd.DataFrame | None] = {}
    if need_preseason:
        for year in years:
            preseason_by_year[year] = load_or_generate_preseason_cache(
                year,
                cache_dir=cache_dir,
                df_featured=df_featured,
                data_config=data_config,
                retrain=args.retrain,
                seed=args.seed,
            )

    # Leave-one-year-out shrinkage tau fit: for each eval year Y, fit on
    # rows from {years} - {Y} so the reported shrinkage metrics aren't
    # optimistically biased by tuning on their own ros_* labels. Needs >= 2
    # eval years; one-year runs fall back to DEFAULT_TAU0.
    shrinkage_tau_by_year: dict[int, dict[str, float] | None] = {}
    if args.fit_shrinkage_tau and "shrinkage" in args.include:
        if len(years) < 2:
            logger.warning(
                "--fit-shrinkage-tau needs >= 2 eval years for leave-one-year-out "
                "cross-fitting; falling back to DEFAULT_TAU0.",
            )
        else:
            for eval_year in years:
                fit_years = [y for y in years if y != eval_year]
                shrinkage_tau_by_year[eval_year] = _fit_shrinkage_tau_from_years(
                    fit_years=fit_years,
                    snapshots=snapshots,
                    thresholds=args.thresholds,
                    cache_dir=cache_dir,
                )
            logger.info(
                "Fitted shrinkage tau (leave-one-year-out): %s",
                shrinkage_tau_by_year,
            )

    # Phase 2 ensemble — one per eval year, cached under phase2_cache_dir.
    # Failing to train/load gracefully skips phase2 without crashing the
    # rest of the benchmark.
    phase2_by_year: dict[int, object | None] = {}
    # Phase 2 scoring needs the SAME raw preseason feature frame the
    # ensemble trained against (age/park/temporal in raw space) — see
    # ``_phase2_eval_preseason_frame`` for the rationale.
    phase2_preseason_by_year: dict[int, pd.DataFrame | None] = {}
    phase2_config: dict | None = None
    if want_phase2:
        phase2_config_path = Path(args.phase2_config)
        if not phase2_config_path.exists():
            logger.warning(
                "phase2 baseline requested but config %s is missing; skipping phase2.",
                phase2_config_path,
            )
        else:
            phase2_config = _load_phase2_config(
                phase2_config_path,
                epochs_override=args.phase2_epochs,
                n_seeds_override=args.phase2_n_seeds,
            )
            raw_dir_path = Path(args.raw_dir)
            for year in years:
                try:
                    phase2_by_year[year] = _load_or_train_phase2_ensemble(
                        eval_year=year,
                        phase2_cache_dir=phase2_cache_dir,
                        phase2_config=phase2_config,
                        raw_dir=raw_dir_path,
                        df_featured=df_featured,
                        retrain=args.retrain,
                    )
                except FileNotFoundError as exc:
                    logger.error(
                        "Skipping phase2 for %d: %s",
                        year,
                        exc,
                    )
                    phase2_by_year[year] = None
                phase2_preseason_by_year[year] = _phase2_eval_preseason_frame(
                    year, df_featured
                )

    year_results: list[dict] = []
    for year in years:
        logger.info("=" * 60)
        logger.info("Year %d", year)
        logger.info("=" * 60)
        result = evaluate_year(
            year=year,
            snapshots=snapshots,
            thresholds=args.thresholds,
            baselines=args.include,
            preseason=preseason_by_year.get(year),
            prior_pa=args.prior_pa,
            min_ros_pa=args.min_ros_pa,
            shrinkage_tau=shrinkage_tau_by_year.get(year),
            phase2_ensemble=phase2_by_year.get(year),
            phase2_config=phase2_config,
            phase2_preseason=phase2_preseason_by_year.get(year),
        )
        year_results.append(result)

    pooled = pool_by_threshold(year_results)
    print_benchmark_results(year_results, pooled)
    save_benchmark_outputs(year_results, pooled, output_dir, pit_plot=args.pit_plot)

    logger.info("ROS benchmark complete.")


if __name__ == "__main__":
    main()

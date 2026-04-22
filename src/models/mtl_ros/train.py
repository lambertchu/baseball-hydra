"""CLI + callable training pipeline for the Phase 2 ROS quantile MTL.

Wires together the Phase 2 building blocks:

* :func:`src.features.in_season.compute_in_season_features` — 24 in-season
  feature columns derived from weekly snapshot rows.
* :func:`src.features.registry.get_feature_names` — canonical preseason
  feature names we optionally concatenate onto the snapshot frame.
* :class:`src.models.mtl_ros.dataset.ROSSnapshotDataset` is built implicitly
  through :func:`compute_sample_weights`; the forecaster itself already owns
  the DataLoader wiring internally so we only feed it raw arrays.
* :class:`src.models.mtl_ros.splits.walk_forward_split` — season-boundary
  train/val/test split.
* :class:`src.models.mtl_ros.model.MTLQuantileEnsembleForecaster` — the
  multi-seed ensemble trained on cutoff-level targets.

The module exposes:

- :func:`train_ros(config, snapshots_df=None, preseason_df=None)` — programmatic
  entry point used by tests and downstream scripts.  When either frame is
  omitted we fall back to the paths in ``config['data']``.
- :func:`main(argv=None)` — argparse CLI with a ``--smoke`` flag for CI.

Design notes
------------
* Sample weights are computed via :func:`compute_sample_weights` (recency ×
  ``sqrt(ros_pa + 1)``) and handed to the forecaster through its
  ``sample_weights`` argument.  Passing them explicitly bypasses the
  season-only recency path inside the forecaster, which is what we want —
  we already include the sqrt-ros_pa term.
* ``min_ytd_pa`` is applied as a row-level filter *before* computing sample
  weights so the mean-normalisation of the weights is computed over the
  same rows the training loop actually sees.
* Feature selection: we start with the in-season features (opt-in group),
  then union in the preseason groups the user enabled.  Missing columns are
  dropped with a WARNING rather than raising, so a partial preseason table
  still trains.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

from src.features.in_season import compute_in_season_features
from src.features.registry import FeatureGroup, get_feature_names
from src.models.mtl_ros.dataset import compute_sample_weights
from src.models.mtl_ros.model import MTLQuantileEnsembleForecaster
from src.models.mtl_ros.splits import SplitConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ROS rate targets — order must match ``MTLQuantileNetwork``'s head indices:
# 0=OBP, 1=SLG, 2=HR/PA, 3=R/PA, 4=RBI/PA, 5=SB/PA.
ROS_RATE_TARGETS: tuple[str, ...] = (
    "ros_obp",
    "ros_slg",
    "ros_hr_per_pa",
    "ros_r_per_pa",
    "ros_rbi_per_pa",
    "ros_sb_per_pa",
)
ROS_PA_TARGET: str = "ros_pa"

_ID_COLS: tuple[str, ...] = ("mlbam_id", "season")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_yaml_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_snapshots_from_paths(
    config: dict,
    seasons: Sequence[int] | None,
) -> pd.DataFrame:
    """Read ``weekly_snapshots_{year}.parquet`` for each requested season."""
    data_cfg = config.get("data", {})
    snapshots_dir = Path(data_cfg.get("snapshots_dir", "data/raw"))
    pattern = data_cfg.get("snapshots_pattern", "weekly_snapshots_{year}.parquet")

    if seasons is None:
        # Default to a reasonable training window; callers can override.
        splits_cfg = config.get("splits", {})
        train_end = splits_cfg.get("train_end_season", 2022)
        val = splits_cfg.get("val_season")
        test = splits_cfg.get("test_season")
        candidates = list(range(2016, train_end + 1))
        if val is not None:
            candidates.append(val)
        if test is not None:
            candidates.append(test)
        seasons = sorted(set(candidates))

    frames: list[pd.DataFrame] = []
    for year in sorted(set(seasons)):
        path = snapshots_dir / pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(
                f"Weekly snapshot file {path} missing. "
                f"Run: uv run python -m src.data.build_snapshots --seasons {year}"
            )
        frames.append(pd.read_parquet(path))
        logger.info("Loaded %s (%d rows)", path, len(frames[-1]))
    if not frames:
        raise ValueError(
            "No snapshot files loaded; check --seasons / data.snapshots_dir"
        )
    return pd.concat(frames, ignore_index=True)


def _load_preseason_from_path(config: dict) -> pd.DataFrame:
    """Load the preseason merged dataset referenced by ``data.preseason_path``."""
    data_cfg = config.get("data", {})
    preseason_path = Path(
        data_cfg.get("preseason_path", "data/merged_batter_data.parquet")
    )
    if not preseason_path.exists():
        raise FileNotFoundError(
            f"Preseason merged dataset missing at {preseason_path}. "
            f"Run: uv run python -m src.data.merge"
        )
    logger.info("Loading preseason features from %s", preseason_path)
    return pd.read_parquet(preseason_path)


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def _preseason_group_flags(config: dict) -> dict[str, bool]:
    """Translate config's ``data.preseason_feature_groups`` into a registry map.

    The registry keys all registered groups by :class:`FeatureGroup` value
    (e.g. ``"age"``, ``"temporal"``).  We default every group to ``False``
    except the ones flipped on in config, plus ``in_season`` which is keyed
    off ``data.include_in_season_features``.
    """
    data_cfg = config.get("data", {})
    requested = dict(data_cfg.get("preseason_feature_groups", {}))
    flags: dict[str, bool] = {g.value: False for g in FeatureGroup}
    for group_name, enabled in requested.items():
        if group_name not in flags:
            logger.warning(
                "Unknown preseason_feature_groups entry %r; ignoring.", group_name
            )
            continue
        flags[group_name] = bool(enabled)
    flags[FeatureGroup.IN_SEASON.value] = bool(
        data_cfg.get("include_in_season_features", True)
    )
    return flags


def _select_feature_columns(
    joined: pd.DataFrame,
    config: dict,
) -> list[str]:
    """Pick registered feature columns that actually exist on ``joined``.

    Emits a WARNING for each expected-but-absent feature.  Keeps the
    canonical registry order so downstream save/load is stable.
    """
    flags = _preseason_group_flags(config)
    requested = get_feature_names(enabled_groups=flags)
    present = [c for c in requested if c in joined.columns]
    missing = [c for c in requested if c not in joined.columns]
    if missing:
        logger.warning(
            "Dropping %d requested feature(s) not present in joined snapshot + "
            "preseason frame: %s",
            len(missing),
            missing,
        )
    if not present:
        raise ValueError(
            "No usable feature columns after filtering; check "
            "data.preseason_feature_groups and in-season feature builder."
        )
    return present


def _join_preseason(
    snapshots_with_in_season: pd.DataFrame,
    preseason: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join per-week snapshots with one-row-per-player-season preseason.

    Preseason columns collide with in-season columns on IDs only; we rename
    preseason duplicates with a ``_preseason`` suffix and drop them after
    the join so the final frame keeps a single ``mlbam_id`` / ``season``.
    """
    if preseason is None or preseason.empty:
        return snapshots_with_in_season
    # Preseason must carry the join keys.
    missing_keys = [k for k in _ID_COLS if k not in preseason.columns]
    if missing_keys:
        raise KeyError(
            f"Preseason frame missing join keys {missing_keys}; "
            f"expected columns {_ID_COLS}"
        )
    # Join key columns shouldn't be suffixed; everything else can collide.
    overlap = (set(snapshots_with_in_season.columns) & set(preseason.columns)) - set(
        _ID_COLS
    )
    if overlap:
        logger.info(
            "Preseason columns overlap with snapshot columns; dropping preseason "
            "copies: %s",
            sorted(overlap),
        )
        preseason = preseason.drop(columns=list(overlap))
    return snapshots_with_in_season.merge(
        preseason,
        on=list(_ID_COLS),
        how="left",
        validate="many_to_one",
    )


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------


def train_ros(
    config: dict,
    snapshots_df: pd.DataFrame | None = None,
    preseason_df: pd.DataFrame | None = None,
) -> MTLQuantileEnsembleForecaster:
    """Train the Phase 2 ROS quantile MTL ensemble.

    Parameters
    ----------
    config:
        Parsed config dict (see ``configs/mtl_ros.yaml``).
    snapshots_df:
        Optional pre-loaded weekly snapshots.  Loaded from ``config['data']``
        when omitted.  One row per ``(mlbam_id, season, iso_week)`` cutoff.
    preseason_df:
        Optional pre-loaded preseason merged dataset (one row per
        ``(mlbam_id, season)``).  Loaded from ``config['data']`` when omitted;
        pass an empty/None value to skip the preseason join entirely.

    Returns
    -------
    MTLQuantileEnsembleForecaster
        Fitted multi-seed ensemble.  Never saved here — the caller decides
        where to persist via ``MTLQuantileEnsembleForecaster.save()``.
    """
    train_cfg = config.get("training", {})
    min_ytd_pa = int(train_cfg.get("min_ytd_pa", 50))
    recency_lambda = float(train_cfg.get("recency_decay_lambda", 0.30))

    # 1. Load snapshots + preseason.
    if snapshots_df is None:
        snapshots_df = _load_snapshots_from_paths(config, seasons=None)
    else:
        snapshots_df = snapshots_df.copy()
    logger.info("Snapshots: %d rows", len(snapshots_df))

    if preseason_df is None:
        data_cfg = config.get("data", {})
        preseason_path = data_cfg.get("preseason_path")
        # Only load from disk when a path is set; otherwise skip preseason join.
        if preseason_path is not None and Path(preseason_path).exists():
            preseason_df = _load_preseason_from_path(config)
        else:
            logger.info(
                "No preseason frame supplied and %s missing — training on in-season "
                "features only.",
                preseason_path,
            )
            preseason_df = None
    elif preseason_df.empty:
        preseason_df = None

    # 2. Compute in-season features on the snapshot frame.
    in_season = compute_in_season_features(snapshots_df)
    # Drop any duplicate columns in the snapshot frame before concatenating.
    dup_cols = [c for c in in_season.columns if c in snapshots_df.columns]
    snapshots_enriched = pd.concat(
        [snapshots_df.drop(columns=dup_cols, errors="ignore"), in_season],
        axis=1,
    )

    # 3. Join preseason onto the enriched snapshot frame.
    joined = _join_preseason(snapshots_enriched, preseason_df)

    # 4. Filter by min_ytd_pa before the split so split sizes are realistic.
    if min_ytd_pa > 0 and "pa_ytd" in joined.columns:
        before = len(joined)
        joined = joined.loc[joined["pa_ytd"].fillna(0) >= min_ytd_pa].reset_index(
            drop=True
        )
        logger.info(
            "min_ytd_pa=%d filter: %d → %d rows", min_ytd_pa, before, len(joined)
        )

    # 5. Select feature columns (honour the registry + config feature groups).
    feature_cols = _select_feature_columns(joined, config)
    logger.info("Selected %d feature columns", len(feature_cols))

    # 6. Walk-forward split.
    split_cfg = SplitConfig.from_dict(config["splits"])
    splits = split_cfg.build(joined)
    train_frame = splits["train"]
    val_frame = splits.get("val")
    if len(train_frame) == 0:
        raise ValueError(
            "Train split is empty; check splits.train_end_season and that "
            "snapshots cover that range."
        )
    logger.info(
        "Split sizes — train: %d, val: %s",
        len(train_frame),
        len(val_frame) if val_frame is not None else "(none)",
    )

    # 7. Drop rows with NaN ROS rate targets in training — the loss would
    #    otherwise produce NaN gradients.  (Val split keeps NaNs; the loss
    #    mask would just skip them if we cared, but we don't require val.)
    train_frame = _drop_rows_with_nan_targets(train_frame, ROS_RATE_TARGETS)
    if val_frame is not None:
        val_frame = _drop_rows_with_nan_targets(val_frame, ROS_RATE_TARGETS)

    # 8. Materialise training matrices.
    X_train = train_frame[feature_cols]
    y_train = train_frame[list(ROS_RATE_TARGETS)]
    pa_train = np.asarray(
        train_frame[ROS_PA_TARGET].fillna(0.0).clip(lower=0.0).to_numpy(),
        dtype=np.float64,
    )

    # 9. Sample weights (recency × sqrt(ros_pa+1)).
    sample_weights = compute_sample_weights(
        train_frame,
        recency_lambda=recency_lambda,
        ros_pa_col=ROS_PA_TARGET,
        season_col="season",
    )

    eval_set = None
    if val_frame is not None and len(val_frame) > 0:
        X_val = val_frame[feature_cols]
        y_val = val_frame[list(ROS_RATE_TARGETS)]
        pa_val = np.asarray(
            val_frame[ROS_PA_TARGET].fillna(0.0).clip(lower=0.0).to_numpy(),
            dtype=np.float64,
        )
        eval_set = (X_val, y_val, pa_val)

    # 10. Train the ensemble.
    ensemble = MTLQuantileEnsembleForecaster(config)
    ensemble.fit(
        X_train,
        y_train,
        pa_target=pa_train,
        sample_weights=sample_weights,
        eval_set=eval_set,
    )
    logger.info(
        "Trained %d-seed ROS ensemble on %d cutoff rows",
        len(ensemble.forecasters_),
        len(X_train),
    )
    return ensemble


def _drop_rows_with_nan_targets(
    frame: pd.DataFrame,
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Drop rows whose rate targets are all NaN (and clamp per-target NaN elsewhere).

    Rationale: the quantile loss currently averages over all batch rows, so
    a fully-NaN target row would poison every task.  Partial NaNs (e.g. a
    player with no SB in ROS but valid OBP) are rare in real data; we
    conservatively drop them here too rather than risk NaN gradients.
    """
    present = [c for c in target_cols if c in frame.columns]
    if not present:
        return frame
    mask = frame[present].notna().all(axis=1)
    dropped = (~mask).sum()
    if dropped > 0:
        logger.info(
            "Dropping %d/%d rows with missing ROS rate targets",
            int(dropped),
            len(frame),
        )
    return frame.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_season_range(token: str) -> list[int]:
    """Accept ``2016-2022`` or comma-separated ``2020,2021,2023`` tokens."""
    token = token.strip()
    if "-" in token:
        lo, hi = token.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in token.split(",") if s.strip()]


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Phase 2 ROS quantile MTL ensemble.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mtl_ros.yaml",
        help="Path to the mtl_ros YAML config.",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=None,
        help="Optional season filter, e.g. '2016-2022' or '2020,2021'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu/cuda); updates config['training']['device'].",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for the saved ensemble (overrides output.model_dir).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a minimal training pass against in-memory synthetic data. "
            "Ignores --seasons; useful for CI."
        ),
    )
    return parser


def _make_smoke_fixtures() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tiny synthetic dataset + config for ``--smoke``."""
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for season in (2020, 2021, 2022, 2023):
        for pid in (101, 102, 103, 104, 105):
            for wk in (20, 24, 28):
                rows.append(
                    {
                        "mlbam_id": pid,
                        "season": season,
                        "iso_year": season,
                        "iso_week": int(wk),
                        "pa_ytd": float(60.0 + rng.uniform(0.0, 300.0)),
                        "obp_ytd": float(rng.uniform(0.28, 0.40)),
                        "slg_ytd": float(rng.uniform(0.35, 0.50)),
                        "hr_per_pa_ytd": float(rng.uniform(0.01, 0.06)),
                        "r_per_pa_ytd": float(rng.uniform(0.08, 0.15)),
                        "rbi_per_pa_ytd": float(rng.uniform(0.08, 0.15)),
                        "sb_per_pa_ytd": float(rng.uniform(0.0, 0.03)),
                        "iso_ytd": float(rng.uniform(0.10, 0.25)),
                        "bb_rate_ytd": float(rng.uniform(0.06, 0.14)),
                        "k_rate_ytd": float(rng.uniform(0.15, 0.30)),
                        "trail4w_pa": float(rng.uniform(40.0, 100.0)),
                        "trail4w_h": float(rng.uniform(10.0, 30.0)),
                        "trail4w_bb": float(rng.uniform(2.0, 10.0)),
                        "trail4w_hbp": float(rng.uniform(0.0, 3.0)),
                        "trail4w_sf": float(rng.uniform(0.0, 2.0)),
                        "trail4w_ab": float(rng.uniform(30.0, 90.0)),
                        "trail4w_singles": float(rng.uniform(6.0, 20.0)),
                        "trail4w_doubles": float(rng.uniform(2.0, 8.0)),
                        "trail4w_triples": float(rng.uniform(0.0, 2.0)),
                        "trail4w_hr": float(rng.uniform(0.0, 5.0)),
                        "trail4w_r": float(rng.uniform(4.0, 15.0)),
                        "trail4w_rbi": float(rng.uniform(4.0, 15.0)),
                        "trail4w_sb": float(rng.uniform(0.0, 2.0)),
                        "trail4w_so": float(rng.uniform(6.0, 25.0)),
                        "ros_pa": float(rng.uniform(50.0, 350.0)),
                        "ros_obp": float(rng.uniform(0.28, 0.40)),
                        "ros_slg": float(rng.uniform(0.35, 0.50)),
                        "ros_hr_per_pa": float(rng.uniform(0.01, 0.06)),
                        "ros_r_per_pa": float(rng.uniform(0.08, 0.15)),
                        "ros_rbi_per_pa": float(rng.uniform(0.08, 0.15)),
                        "ros_sb_per_pa": float(rng.uniform(0.0, 0.03)),
                    }
                )
    snapshots = pd.DataFrame(rows)
    keys = snapshots[["mlbam_id", "season"]].drop_duplicates().reset_index(drop=True)
    preseason = pd.DataFrame(
        {
            "mlbam_id": keys["mlbam_id"],
            "season": keys["season"],
            "age": rng.uniform(22.0, 34.0, len(keys)),
            "age_squared": rng.uniform(484.0, 1156.0, len(keys)),
            "park_factor_runs": rng.uniform(95.0, 110.0, len(keys)),
            "park_factor_hr": rng.uniform(90.0, 115.0, len(keys)),
            "weighted_avg_obp": rng.uniform(0.28, 0.40, len(keys)),
            "weighted_avg_slg": rng.uniform(0.35, 0.50, len(keys)),
            "team_runs_per_game": rng.uniform(3.8, 5.5, len(keys)),
            "team_ops": rng.uniform(0.680, 0.800, len(keys)),
        }
    )
    config = {
        "model": {
            "n_quantiles": 5,
            "hidden_dims": [16, 8],
            "head_dim": 4,
            "dropouts": [0.1, 0.1],
            "two_stage": True,
            "speed_head_indices": [5],
            "taus": [0.05, 0.25, 0.50, 0.75, 0.95],
        },
        "loss": {"pa_loss": "mse", "pa_weight": 1.0},
        "training": {
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "early_stopping_patience": 2,
            "recency_decay_lambda": 0.30,
            "min_ytd_pa": 50,
            "device": "cpu",
        },
        "ensemble": {"n_seeds": 1, "base_seed": 42},
        "data": {
            "preseason_feature_groups": {
                "age": True,
                "park_factors": True,
                "team_stats": True,
                "temporal": True,
            },
            "include_in_season_features": True,
        },
        "splits": {
            "train_end_season": 2022,
            "val_season": 2023,
        },
        "output": {"model_dir": "data/models/mtl_ros_quantile_smoke"},
        "seed": 42,
    }
    return snapshots, preseason, config


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry-point.  Returns the exit code (0 on success)."""
    parser = _build_argparser()
    args = parser.parse_args(argv)

    if args.smoke:
        snapshots, preseason, config = _make_smoke_fixtures()
        if args.device is not None:
            config.setdefault("training", {})["device"] = args.device
        if args.out is not None:
            config.setdefault("output", {})["model_dir"] = args.out
        ensemble = train_ros(config, snapshots_df=snapshots, preseason_df=preseason)
        out_dir = Path(config["output"]["model_dir"])
        ensemble.save(out_dir)
        logger.info("Smoke train complete → %s", out_dir)
        return 0

    config = _load_yaml_config(args.config)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.out is not None:
        config.setdefault("output", {})["model_dir"] = args.out

    seasons: list[int] | None = None
    if args.seasons is not None:
        seasons = _parse_season_range(args.seasons)

    snapshots = _load_snapshots_from_paths(config, seasons=seasons)
    preseason_path = config.get("data", {}).get("preseason_path")
    if preseason_path is not None and Path(preseason_path).exists():
        preseason = _load_preseason_from_path(config)
    else:
        logger.warning(
            "Preseason path %s missing; continuing without preseason features.",
            preseason_path,
        )
        preseason = None

    ensemble = train_ros(config, snapshots_df=snapshots, preseason_df=preseason)
    out_dir = Path(
        config.get("output", {}).get("model_dir", "data/models/mtl_ros_quantile")
    )
    ensemble.save(out_dir)
    logger.info("Saved ROS ensemble → %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via tests via main()
    raise SystemExit(main())

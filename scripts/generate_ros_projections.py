"""Generate rest-of-season (ROS) projections for the current season.

Trains a Phase 2 ROS quantile MTL ensemble on all historical snapshots
(seasons strictly before ``--year``) plus the preseason features from
the merged preseason dataset, then predicts ROS rates + remaining PA
for the latest available week per player in ``--year``'s snapshot.

Output: CSV at ``data/projections/ros_mtl_{year}.csv`` with columns
``mlbam_id``, ``season``, ``iso_week``, ``pa_ytd``, ``ros_pa_pred`` plus
per-target median + lower/upper quantile columns.

Because the merged preseason dataset only runs through ``year - 1``,
we synthesize a ``year`` preseason row per player by copying their
``year - 1`` row, bumping ``age`` by 1, and dropping ``target_*``.
Temporal features (prev_year_*, weighted_avg_*, trend_*) thus use the
``year - 1`` values, which is a small approximation — rebuild with a
full merge once FG batting for the current year is fetchable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.in_season import compute_in_season_features
from src.features.registry import PLAYER_SEASON_KEY, FeatureGroup, get_feature_names
from src.models.baselines.shrinkage import ROS_RATE_TARGETS
from src.models.mtl_ros.train import train_ros

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_historical_snapshots(
    raw_dir: Path, year: int
) -> pd.DataFrame:
    """Concat all ``weekly_snapshots_{y}.parquet`` for y < year."""
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("weekly_snapshots_*.parquet")):
        try:
            y = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if y < year:
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(
            f"No historical snapshots for years < {year} at {raw_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def _synthesize_current_preseason(
    merged: pd.DataFrame, year: int
) -> pd.DataFrame:
    """Make a ``year`` preseason frame by copying ``year - 1`` rows.

    age += 1, age_squared recomputed, season reassigned, target_* dropped.
    Temporal features (weighted_avg/*, prev_year/*, trend/*) are left as
    their ``year - 1`` values — this is the approximation called out in
    the module docstring.
    """
    prev = merged.loc[merged["season"] == year - 1].copy()
    if prev.empty:
        raise ValueError(
            f"Merged dataset has no rows for season {year - 1}; "
            "cannot synthesize preseason features for ROS inference."
        )
    prev["season"] = year
    if "age" in prev.columns:
        prev["age"] = prev["age"].astype("Int64") + 1
    if "age_squared" in prev.columns:
        prev["age_squared"] = prev["age"].astype(float) ** 2
    # target_* are Y+1 stats; not needed for inference on year Y.
    drop_cols = [c for c in prev.columns if c.startswith("target_")]
    prev = prev.drop(columns=drop_cols, errors="ignore")
    return prev.reset_index(drop=True)


def _feature_flags_from_config(config: dict) -> dict[str, bool]:
    data_cfg = config.get("data", {})
    flags: dict[str, bool] = {g.value: False for g in FeatureGroup}
    for g, enabled in data_cfg.get("preseason_feature_groups", {}).items():
        if g in flags:
            flags[g] = bool(enabled)
    flags[FeatureGroup.IN_SEASON.value] = bool(
        data_cfg.get("include_in_season_features", True)
    )
    return flags


def _latest_row_per_player(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (mlbam_id, season): the snapshot with the max
    iso_week (i.e. the most recent week the player qualifies for, with
    pa_ytd >= min filter already applied by build_snapshots)."""
    key_cols = list(PLAYER_SEASON_KEY)
    idx = snapshots.groupby(key_cols)["iso_week"].idxmax()
    return snapshots.loc[idx].reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/mtl_ros.yaml")
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw")
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=Path("data/merged_batter_data.parquet"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="CSV output path (default: data/projections/ros_mtl_{year}.csv)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=None, help="Override ensemble.n_seeds"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training.epochs"
    )
    parser.add_argument("--device", default=None, help="cpu / cuda / mps")
    args = parser.parse_args(argv)

    import yaml

    with open(args.config) as f:
        config: dict = yaml.safe_load(f)
    if args.n_seeds is not None:
        config.setdefault("ensemble", {})["n_seeds"] = int(args.n_seeds)
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = int(args.epochs)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    # Force splits so EVERY historical row lands in train.
    config["splits"] = {"train_end_season": int(args.year - 1)}

    # --- Training data --------------------------------------------------
    snapshots_train = _load_historical_snapshots(args.raw_dir, args.year)
    logger.info(
        "Historical training snapshots: %d rows across seasons %s",
        len(snapshots_train),
        sorted(snapshots_train["season"].unique()),
    )
    merged = pd.read_parquet(args.merged_path)
    logger.info(
        "Merged preseason: %d rows across seasons %s",
        len(merged),
        sorted(merged["season"].unique()),
    )
    synth_current = _synthesize_current_preseason(merged, args.year)
    logger.info(
        "Synthesized preseason for %d: %d rows (derived from %d-1 rows)",
        args.year,
        len(synth_current),
        args.year,
    )
    preseason_full = pd.concat([merged, synth_current], ignore_index=True)

    # --- Train ----------------------------------------------------------
    ensemble = train_ros(
        config, snapshots_df=snapshots_train, preseason_df=preseason_full
    )
    logger.info(
        "Trained %d-seed ensemble on %d feature columns",
        len(getattr(ensemble, "forecasters_", [])),
        len(getattr(ensemble, "feature_names_", [])),
    )

    # --- Inference on latest 2026 snapshot per player -------------------
    snap_path = args.raw_dir / f"weekly_snapshots_{args.year}.parquet"
    if not snap_path.exists():
        raise FileNotFoundError(
            f"Current-year snapshot missing at {snap_path}; "
            f"run: python -m src.data.build_snapshots --seasons {args.year}"
        )
    current = pd.read_parquet(snap_path)
    latest = _latest_row_per_player(current)
    logger.info("Latest-week rows for %d: %d players", args.year, len(latest))

    # Compute in-season features + join synthesized preseason
    in_season_feats = compute_in_season_features(latest)
    dup_cols = [c for c in in_season_feats.columns if c in latest.columns]
    enriched = pd.concat(
        [latest.drop(columns=dup_cols, errors="ignore"), in_season_feats],
        axis=1,
    )
    overlap = (set(enriched.columns) & set(synth_current.columns)) - set(
        PLAYER_SEASON_KEY
    )
    pre_to_join = synth_current.drop(columns=list(overlap), errors="ignore")
    enriched = enriched.merge(
        pre_to_join,
        on=list(PLAYER_SEASON_KEY),
        how="left",
        validate="many_to_one",
    )

    feature_names = list(ensemble.feature_names_)
    X = enriched.reindex(columns=feature_names).astype(float)
    # Missing-feature fill using training means (neutral post-scaling)
    train_names = list(ensemble.forecasters_[0].feature_names_)
    scaler = ensemble.forecasters_[0].feature_scaler_
    mean_by_name = dict(zip(train_names, scaler.mean_))
    X.fillna(value=mean_by_name, inplace=True)
    X.fillna(value=0.0, inplace=True)

    preds = ensemble.predict(X)
    q = np.asarray(preds["quantiles"], dtype=np.float64)  # (n, 6, 5)
    pa_pred = np.asarray(preds["pa_remaining"], dtype=np.float64).squeeze(-1)

    taus = list(config["model"].get("taus", [0.05, 0.25, 0.50, 0.75, 0.95]))
    median_idx = taus.index(0.50)
    lo_idx = taus.index(0.05) if 0.05 in taus else 0
    hi_idx = taus.index(0.95) if 0.95 in taus else len(taus) - 1

    # --- Assemble CSV ---------------------------------------------------
    out_rows: list[dict] = []
    for row_i, (_, row) in enumerate(latest.iterrows()):
        record = {
            "mlbam_id": int(row["mlbam_id"]),
            "season": int(row["season"]),
            "iso_week": int(row["iso_week"]),
            "pa_ytd": float(row.get("pa_ytd", np.nan)),
            "ros_pa_pred": float(max(pa_pred[row_i], 0.0)),
        }
        for t_idx, stat in enumerate(ROS_RATE_TARGETS):
            col = stat  # e.g. ros_obp, ros_hr_per_pa
            record[f"{col}_p50"] = float(q[row_i, t_idx, median_idx])
            record[f"{col}_p05"] = float(q[row_i, t_idx, lo_idx])
            record[f"{col}_p95"] = float(q[row_i, t_idx, hi_idx])
        # Convenience: ROS counts for HR/R/RBI/SB = per-PA rate * ROS PA
        ros_pa = max(record["ros_pa_pred"], 0.0)
        for stat, ros_col in zip(
            ("hr", "r", "rbi", "sb"),
            ("ros_hr_per_pa", "ros_r_per_pa", "ros_rbi_per_pa", "ros_sb_per_pa"),
        ):
            record[f"ros_{stat}_count_p50"] = round(
                record[f"{ros_col}_p50"] * ros_pa, 1
            )
        out_rows.append(record)

    out_df = pd.DataFrame(out_rows)
    out_path = args.out or Path(f"data/projections/ros_mtl_{args.year}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote %d rows -> %s", len(out_df), out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

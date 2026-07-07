"""Training CLI for the Phase 3 sequential ROS model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from src.eval.ros_metrics import ROS_RATE_TARGETS
from src.features.in_season import compute_in_season_features
from src.features.registry import PLAYER_SEASON_KEY
from src.models.mtl_ros.model import (
    MTLQuantileEnsembleForecaster,
    MTLQuantileForecaster,
)
from src.models.mtl_ros.splits import SplitConfig
from src.models.mtl_ros.train import _make_smoke_fixtures, train_ros
from src.models.ros.dataset import default_sequence_sample_weights
from src.models.ros.model import ROSSequenceEnsembleForecaster

logger = logging.getLogger(__name__)

ROS_PA_TARGET = "ros_pa"


def _load_yaml_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_season_range(token: str) -> list[int]:
    token = token.strip()
    if "-" in token:
        lo, hi = token.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in token.split(",") if s.strip()]


def _load_snapshots_from_paths(
    config: dict, seasons: Sequence[int] | None
) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("snapshots_dir", "data/raw"))
    pattern = data_cfg.get("snapshots_pattern", "weekly_snapshots_{year}.parquet")
    if seasons is None:
        splits = config.get("splits", {})
        train_end = int(splits.get("train_end_season", 2022))
        candidates = list(range(2016, train_end + 1))
        for key in ("val_season", "test_season"):
            if splits.get(key) is not None:
                candidates.append(int(splits[key]))
        seasons = sorted(set(candidates))
    frames: list[pd.DataFrame] = []
    for year in sorted(set(seasons)):
        path = raw_dir / pattern.format(year=year)
        if not path.exists():
            raise FileNotFoundError(
                f"Weekly snapshot file {path} missing. Backfill Phase 3 data with: "
                "uv run python -m src.data.fetch_game_logs --seasons 2016-2022 && "
                "uv run python -m src.data.fetch_raw_statcast --seasons 2016-2022 && "
                "uv run python -m src.data.build_snapshots --seasons 2016-2022"
            )
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("No weekly snapshots loaded.")
    return pd.concat(frames, ignore_index=True)


def _load_preseason_from_path(config: dict) -> pd.DataFrame | None:
    path = config.get("data", {}).get("preseason_path")
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _enforce_backfill_gate(config: dict, snapshots: pd.DataFrame) -> None:
    data_cfg = config.get("data", {})
    if not bool(data_cfg.get("enforce_backfill_gate", True)):
        return
    min_years = int(data_cfg.get("min_snapshot_years", 7))
    seasons = sorted(int(s) for s in snapshots["season"].dropna().unique())
    if len(seasons) < min_years:
        raise ValueError(
            f"Phase 3 requires at least {min_years} historical weekly snapshot "
            f"seasons; found {len(seasons)} ({seasons}). Backfill with: "
            "uv run python -m src.data.fetch_game_logs --seasons 2016-2022; "
            "uv run python -m src.data.fetch_raw_statcast --seasons 2016-2022; "
            "uv run python -m src.data.build_snapshots --seasons 2016-2022"
        )


def load_phase2_seed0(path: str | Path) -> MTLQuantileForecaster:
    """Load a Phase 2 single-seed forecaster from a seed dir or ensemble dir."""
    p = Path(path)
    if (p / "mtl_ros_forecaster.pt").exists():
        return MTLQuantileForecaster.load(p)
    if (p / "seed_0" / "mtl_ros_forecaster.pt").exists():
        return MTLQuantileForecaster.load(p / "seed_0")
    if (p / "ensemble_meta.json").exists():
        ens = MTLQuantileEnsembleForecaster.load(p)
        return ens.forecasters_[0]
    raise FileNotFoundError(
        f"Could not find a Phase 2 seed_0 checkpoint under {p}. "
        "Train one with: uv run python -m src.models.mtl_ros.train "
        "--config configs/mtl_ros.yaml"
    )


def build_phase2_feature_frame(
    rows: pd.DataFrame,
    base_forecaster: MTLQuantileForecaster,
    preseason: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build raw Phase 2 feature rows in the base forecaster's feature order."""
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
    if preseason is not None and not preseason.empty:
        overlap = (set(enriched.columns) & set(preseason.columns)) - set(
            PLAYER_SEASON_KEY
        )
        pre = preseason.drop(columns=list(overlap), errors="ignore")
        enriched = enriched.merge(
            pre,
            on=list(PLAYER_SEASON_KEY),
            how="left",
            validate="many_to_one",
        )
    names = list(base_forecaster.feature_names_)
    X = enriched.reindex(columns=names).astype(float)
    means = dict(zip(names, base_forecaster.feature_scaler_.mean_))
    X.fillna(value=means, inplace=True)
    X.fillna(value=0.0, inplace=True)
    return X


def _drop_nan_targets(frame: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in ROS_RATE_TARGETS if c in frame.columns]
    mask = frame[present].notna().all(axis=1)
    return frame.loc[mask].reset_index(drop=True)


def train_ros_sequence(
    config: dict,
    snapshots_df: pd.DataFrame | None = None,
    preseason_df: pd.DataFrame | None = None,
    base_forecaster: MTLQuantileForecaster | None = None,
) -> ROSSequenceEnsembleForecaster:
    """Train a Phase 3 GRU ensemble."""
    if snapshots_df is None:
        snapshots_df = _load_snapshots_from_paths(config, seasons=None)
    else:
        snapshots_df = snapshots_df.copy()
    if preseason_df is None:
        preseason_df = _load_preseason_from_path(config)
    _enforce_backfill_gate(config, snapshots_df)

    if base_forecaster is None:
        base_path = config.get("base_phase2", {}).get("model_dir")
        if not base_path:
            raise ValueError("configs/ros.yaml must set base_phase2.model_dir")
        base_forecaster = load_phase2_seed0(base_path)

    train_cfg = config.get("training", {})
    min_ytd_pa = int(train_cfg.get("min_ytd_pa", 50))
    if min_ytd_pa > 0 and "pa_ytd" in snapshots_df.columns:
        snapshots_df = snapshots_df.loc[
            snapshots_df["pa_ytd"].fillna(0) >= min_ytd_pa
        ].reset_index(drop=True)

    split_cfg = SplitConfig.from_dict(config["splits"])
    splits = split_cfg.build(snapshots_df)
    train_frame = _drop_nan_targets(splits["train"])
    val_frame = _drop_nan_targets(splits["val"]) if "val" in splits else None
    if len(train_frame) == 0:
        raise ValueError("No usable Phase 3 training rows after filtering.")

    X_train = build_phase2_feature_frame(train_frame, base_forecaster, preseason_df)
    y_train = train_frame[list(ROS_RATE_TARGETS)]
    pa_train = train_frame[ROS_PA_TARGET].fillna(0.0).clip(lower=0.0)
    weights = default_sequence_sample_weights(
        train_frame,
        recency_lambda=float(train_cfg.get("recency_decay_lambda", 0.30)),
    )

    eval_set = None
    if val_frame is not None and len(val_frame) > 0:
        X_val = build_phase2_feature_frame(val_frame, base_forecaster, preseason_df)
        eval_set = (
            val_frame,
            X_val,
            val_frame[list(ROS_RATE_TARGETS)],
            val_frame[ROS_PA_TARGET].fillna(0.0).clip(lower=0.0),
        )

    ensemble = ROSSequenceEnsembleForecaster(base_forecaster, config)
    ensemble.fit(
        train_frame,
        X_train,
        y_train,
        pa_train,
        sample_weights=weights,
        eval_set=eval_set,
    )
    return ensemble


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Phase 3 ROS GRU ensemble.")
    parser.add_argument("--config", default="configs/ros.yaml")
    parser.add_argument("--seasons", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_argparser().parse_args(argv)
    config = _load_yaml_config(args.config)
    if args.device is not None:
        config.setdefault("training", {})["device"] = args.device
    if args.out is not None:
        config.setdefault("output", {})["model_dir"] = args.out

    if args.smoke:
        snapshots, preseason, phase2_config = _make_smoke_fixtures()
        phase2_config["ensemble"]["n_seeds"] = 1
        phase2_config["training"]["epochs"] = 1
        phase2_config["training"]["batch_size"] = 8
        base_ensemble = train_ros(
            phase2_config,
            snapshots_df=snapshots,
            preseason_df=preseason,
        )
        config.setdefault("data", {})["enforce_backfill_gate"] = False
        config.setdefault("splits", {"train_end_season": 2022, "val_season": 2023})
        config.setdefault("ensemble", {})["n_seeds"] = 1
        config.setdefault("training", {})["epochs"] = 1
        config["training"]["batch_size"] = 8
        ensemble = train_ros_sequence(
            config,
            snapshots_df=snapshots,
            preseason_df=preseason,
            base_forecaster=base_ensemble.forecasters_[0],
        )
    else:
        seasons = _parse_season_range(args.seasons) if args.seasons else None
        snapshots = _load_snapshots_from_paths(config, seasons=seasons)
        preseason = _load_preseason_from_path(config)
        ensemble = train_ros_sequence(
            config,
            snapshots_df=snapshots,
            preseason_df=preseason,
            base_forecaster=None,
        )

    out_dir = Path(
        config.get("output", {}).get("model_dir", "data/models/ros_sequence")
    )
    ensemble.save(out_dir)
    logger.info("Saved Phase 3 ROS ensemble -> %s", out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

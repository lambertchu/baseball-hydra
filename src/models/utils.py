"""Shared utilities for model classes."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def get_model_configs() -> dict[str, dict]:
    """Return the model configuration registry.

    Lazy import avoids circular dependency with ``src.models.mtl.model``.
    """
    from src.models.mtl.model import MTLForecaster

    return {
        "mtl": {
            "class": MTLForecaster,
            "config_path": "configs/mtl.yaml",
            "model_dir": "data/models/mtl/",
            "display_name": "MTL",
        },
    }


def train_model_for_year(
    model_key: str,
    retrain_df: pd.DataFrame,
    data_config: dict,
    seed: int = 42,
) -> object:
    """Train a fresh model on ``retrain_df`` for one evaluation fold.

    When ``retrain_df`` spans multiple seasons, the most recent one is carved
    out as the early-stopping evaluation set; otherwise training runs without
    an eval set. Returns the fit model instance.
    """
    from src.features.pipeline import extract_xy
    from src.models.mtl.model import MTLEnsembleForecaster, MTLForecaster

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

    info = get_model_configs()[model_key]
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


def align_features(X: pd.DataFrame, model: object, model_name: str = "") -> pd.DataFrame:
    """Subset and reorder X to match the features a model was trained with."""
    if not hasattr(model, "feature_names_") or not model.feature_names_:
        return X
    expected = model.feature_names_
    extra = sorted(set(X.columns) - set(expected))
    missing = sorted(set(expected) - set(X.columns))
    if extra or missing:
        label = f" for {model_name}" if model_name else ""
        logger.warning(
            "Feature alignment%s: %d expected, %d provided, %d extra dropped, %d missing filled with 0",
            label, len(expected), len(X.columns), len(extra), len(missing),
        )
    data = {feat: X[feat] if feat in X.columns else 0.0 for feat in expected}
    return pd.DataFrame(data, index=X.index)


def to_float64_array(data: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert input to float64 ndarray, handling nullable pandas dtypes.

    Pandas nullable types (``Float64``, ``Int64``) use ``pd.NA`` instead of
    ``np.nan``.  ``np.asarray(..., dtype=np.float64)`` raises ``TypeError``
    on ``pd.NA``, so we convert via ``.astype()`` first.
    """
    if isinstance(data, pd.DataFrame):
        data = data.astype(np.float64)
    return np.asarray(data, dtype=np.float64)

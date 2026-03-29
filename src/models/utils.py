"""Shared utilities for model classes."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

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

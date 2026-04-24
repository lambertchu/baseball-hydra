"""Per-row sample-weight helper for ROS snapshot training.

The production training path (``src.models.mtl_ros.train.train_ros``) wraps
its own tensors via ``MTLQuantileForecaster``'s internal ``_QuantileDataset``.
This module exposes :func:`compute_sample_weights`, the recency ×
``sqrt(ros_pa + 1)`` weighting used to down-weight noisy tiny-denominator
rows and up-weight long-horizon ones before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sample_weights(
    snapshots: pd.DataFrame,
    recency_lambda: float = 0.30,
    ros_pa_col: str = "ros_pa",
    season_col: str = "season",
) -> np.ndarray:
    """Build per-row sample weights for ROS training.

    Weight formula:

        raw = exp(-lambda * (max_season - season)) * sqrt(ros_pa + 1.0)
        weight = raw / mean(raw)

    The ``sqrt(ros_pa+1)`` term down-weights rows with small ROS denominators
    (rate targets are noisy when only a few PA remain) and up-weights longer
    horizons.  The ``+1`` keeps weights finite and positive on the final-week
    rows where ``ros_pa == 0``.  The mean-normalization keeps the overall
    loss scale comparable to uniform weighting.

    Parameters
    ----------
    snapshots:
        DataFrame with at least ``season_col`` and ``ros_pa_col``.  Weights
        are returned in the same row order as the input frame.
    recency_lambda:
        Decay rate for the season term.  ``0.0`` disables recency decay and
        returns pure ``sqrt(ros_pa+1)`` weights; larger values down-weight
        older seasons more aggressively.  Default 0.30 matches the preseason
        MTL's recency lambda.

    Returns
    -------
    np.ndarray of shape ``(len(snapshots),)`` and dtype float32.
    """
    if len(snapshots) == 0:
        return np.zeros(0, dtype=np.float32)

    season = snapshots[season_col].to_numpy(dtype=np.float64)
    ros_pa = snapshots[ros_pa_col].to_numpy(dtype=np.float64)
    # ros_pa is a count; guard against NaN just in case.
    ros_pa = np.where(np.isfinite(ros_pa), ros_pa, 0.0)
    ros_pa = np.maximum(ros_pa, 0.0)

    max_season = season.max()
    recency = np.exp(-recency_lambda * (max_season - season))
    horizon = np.sqrt(ros_pa + 1.0)
    raw = recency * horizon

    mean = raw.mean()
    if mean <= 0 or not np.isfinite(mean):
        # Degenerate case: fall back to uniform weights.
        return np.ones(len(snapshots), dtype=np.float32)
    return (raw / mean).astype(np.float32)

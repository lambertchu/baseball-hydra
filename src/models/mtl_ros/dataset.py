"""PyTorch Dataset for weekly ROS snapshots.

Wraps rows from ``data/raw/weekly_snapshots_{year}.parquet`` as a
:class:`torch.utils.data.Dataset`.  Each row is one ``(player, season, ISO week)``
cutoff; features describe what we know as of the cutoff and targets describe
the remaining-season outcome.  Training at the cutoff level lets the ROS MTL
observe every point in season-trajectory space instead of just the preseason.

Key contracts
-------------
* Rows with ``pa_ytd < min_ytd_pa`` are filtered out before tensor creation.
  Rate-based targets are too noisy at tiny denominators.
* NaN entries in ``feature_cols`` become 0.0, matching preseason
  ``BatterDataset``.
* ``__getitem__`` returns a dict (not a tuple) to keep the 7-tensor bundle
  ``(x, y, pa_target, weight)`` unambiguous downstream.
* :func:`compute_sample_weights` combines season-level recency decay with a
  ``sqrt(ros_pa + 1)`` term that down-weights tiny ROS denominators and
  up-weights long horizons, then mean-normalizes so weighted and unweighted
  loss magnitudes are comparable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_DEFAULT_RATE_TARGETS: tuple[str, ...] = (
    "ros_obp",
    "ros_slg",
    "ros_hr_per_pa",
    "ros_r_per_pa",
    "ros_rbi_per_pa",
    "ros_sb_per_pa",
)


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


class ROSSnapshotDataset(Dataset):
    """Per-cutoff dataset of (features, rate targets, PA target, weight).

    Parameters
    ----------
    snapshots:
        Weekly snapshot rows.  Must include ``feature_cols``,
        ``rate_target_cols``, ``pa_target_col``, and (implicitly) ``pa_ytd``
        for the ``min_ytd_pa`` filter.
    feature_cols:
        Ordered feature names to emit as the ``x`` tensor.
    rate_target_cols:
        Ordered rate targets (default: the six ROS rate targets).
    pa_target_col:
        Column holding the ROS PA target (default ``"ros_pa"``).
    sample_weight_col:
        Optional column of precomputed sample weights.  When ``None``, all
        rows get a uniform weight of 1.0.  Callers wanting the recency ×
        sqrt(ros_pa) weighting should compute it with
        :func:`compute_sample_weights` and attach the column before
        constructing the dataset.
    min_ytd_pa:
        Filter: keep only rows where ``pa_ytd >= min_ytd_pa``.  Set to 0 to
        disable.
    """

    def __init__(
        self,
        snapshots: pd.DataFrame,
        feature_cols: Sequence[str],
        rate_target_cols: Sequence[str] = _DEFAULT_RATE_TARGETS,
        pa_target_col: str = "ros_pa",
        sample_weight_col: str | None = None,
        min_ytd_pa: int = 50,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.rate_target_cols = list(rate_target_cols)
        self.pa_target_col = pa_target_col

        # Apply the min_ytd_pa filter first so tensors line up with the
        # filtered frame.  ``pa_ytd`` may be missing for callers emitting
        # their own subset; treat that as "no filtering".
        if min_ytd_pa > 0 and "pa_ytd" in snapshots.columns:
            mask = snapshots["pa_ytd"].fillna(0) >= min_ytd_pa
            filtered = snapshots.loc[mask].copy()
        else:
            filtered = snapshots.copy()
        filtered = filtered.reset_index(drop=True)
        self.filtered_snapshots = filtered

        # Features: float32, NaN → 0.0 (mirrors BatterDataset).  `nan_to_num`
        # always returns a freshly-allocated writable array, so torch.from_numpy
        # is safe.
        x_arr = filtered.reindex(columns=self.feature_cols).to_numpy(dtype=np.float32)
        x_arr = np.nan_to_num(x_arr, nan=0.0)
        self.X = torch.from_numpy(x_arr)

        # Rate targets: float32; leave NaN in place (loss layer masks them).
        # Copy to guarantee a writable buffer before handing to torch.
        y_arr = np.array(
            filtered.reindex(columns=self.rate_target_cols).to_numpy(dtype=np.float32),
            copy=True,
        )
        self.y = torch.from_numpy(y_arr)

        # PA target: clamp NaN / negative to 0.0 (end-of-season rows have
        # ros_pa=0, which is the correct label for the PA head).
        pa_raw = filtered[pa_target_col].to_numpy(dtype=np.float32)
        pa_arr = np.where(np.isfinite(pa_raw), pa_raw, 0.0)
        pa_arr = np.maximum(pa_arr, 0.0).astype(np.float32, copy=True)
        self.pa_target = torch.from_numpy(pa_arr).unsqueeze(-1)

        if sample_weight_col is not None:
            w_arr = np.array(
                filtered[sample_weight_col].to_numpy(dtype=np.float32), copy=True
            )
        else:
            w_arr = np.ones(len(filtered), dtype=np.float32)
        self.weights = torch.from_numpy(w_arr)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a dict with keys {x, y, pa_target, weight}.

        Shapes: x=(n_features,), y=(n_rate_targets,), pa_target=(1,),
        weight is a scalar tensor (shape ``()``).
        """
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "pa_target": self.pa_target[idx],
            "weight": self.weights[idx],
        }

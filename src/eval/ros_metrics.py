"""Evaluation metrics for rest-of-season (ROS) projections.

Adds quantile metrics (pinball loss, PIT coverage) on top of the existing
``compute_metrics`` utilities, plus per-player PA-checkpoint row selection
from weekly snapshots.

Target conventions
------------------
ROS targets are all rates so they compose across season lengths:

* ``ros_obp``, ``ros_slg`` — rate stats
* ``ros_{hr,r,rbi,sb}_per_pa`` — per-PA rates for count stats

The matching ytd features are ``obp_ytd``, ``slg_ytd``, and
``{hr,r,rbi,sb}_per_pa_ytd``. The matching preseason MTL columns are
``target_{obp,slg,hr,r,rbi,sb}`` (already in per-PA units when
``rate_targets`` is enabled in ``configs/data.yaml``).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.features.registry import RATE_STATS, TARGET_STATS

# Derived from the single source of truth in ``src/features/registry`` to
# keep stat ordering in lock-step with the preseason model.
ROS_TARGET_STATS: tuple[str, ...] = tuple(TARGET_STATS)
ROS_RATE_TARGETS: tuple[str, ...] = tuple(
    f"ros_{s}" if s in RATE_STATS else f"ros_{s}_per_pa"
    for s in TARGET_STATS
)
ROS_YTD_RATES: tuple[str, ...] = tuple(
    f"{s}_ytd" if s in RATE_STATS else f"{s}_per_pa_ytd"
    for s in TARGET_STATS
)
ROS_TARGET_DISPLAY: tuple[str, ...] = tuple(
    s.upper() if s in RATE_STATS else f"{s.upper()}/PA"
    for s in TARGET_STATS
)
DEFAULT_PA_CHECKPOINTS: tuple[int, ...] = (50, 100, 200, 400)
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


def _as_2d(arr: np.ndarray | pd.DataFrame) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    return out.reshape(-1, 1) if out.ndim == 1 else out


def _resolve_target_names(
    y_true: np.ndarray | pd.DataFrame,
    n_targets: int,
    target_names: Sequence[str] | None,
) -> list[str]:
    if target_names is not None:
        if len(target_names) != n_targets:
            raise ValueError(
                f"target_names has {len(target_names)} entries but y_true has {n_targets} targets",
            )
        return list(target_names)
    if isinstance(y_true, pd.DataFrame):
        return list(y_true.columns)
    return [f"target_{i}" for i in range(n_targets)]


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: float,
) -> float:
    """Pinball (quantile) loss for a single quantile level.

    ``L_τ(y, q) = max(τ (y-q), (τ-1)(y-q))``. Lower is better. A well-calibrated
    quantile prediction minimizes the mean pinball at level τ.
    """
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def quantile_loss(
    y_true: np.ndarray | pd.DataFrame,
    y_quantiles: np.ndarray,
    taus: Sequence[float],
    target_names: Sequence[str] | None = None,
) -> dict:
    """Mean pinball loss per (target, tau) and aggregate.

    Parameters
    ----------
    y_true:
        Shape ``(n_samples, n_targets)`` or ``(n_samples,)`` for one target.
    y_quantiles:
        Shape ``(n_samples, n_targets, n_quantiles)`` — quantile predictions
        aligned with ``taus``. One-target inputs may be ``(n_samples, n_quantiles)``.
    taus:
        Quantile levels (values in [0, 1]) matching the last axis of ``y_quantiles``.

    Returns
    -------
    dict
        ::

            {
                "taus": [0.05, 0.25, ...],
                "per_target": {
                    "OBP": {
                        "mean_pinball": float,
                        "per_tau": {0.05: float, 0.25: float, ...},
                    },
                    ...
                },
                "aggregate": {"mean_pinball": float},
            }
    """
    yt = _as_2d(y_true)
    yq = np.asarray(y_quantiles, dtype=np.float64)
    if yq.ndim == 2:
        # (n_samples, n_quantiles) → add target axis
        yq = yq.reshape(yq.shape[0], 1, yq.shape[1])
    if yq.ndim != 3:
        raise ValueError(
            f"y_quantiles must be 2D or 3D, got shape {yq.shape}",
        )
    if yq.shape[0] != yt.shape[0] or yq.shape[1] != yt.shape[1]:
        raise ValueError(
            f"y_quantiles shape {yq.shape} incompatible with y_true shape {yt.shape}",
        )
    if yq.shape[2] != len(taus):
        raise ValueError(
            f"y_quantiles has {yq.shape[2]} quantiles but {len(taus)} taus",
        )

    names = _resolve_target_names(y_true, yt.shape[1], target_names)
    taus_list = [float(t) for t in taus]
    taus_arr = np.asarray(taus_list, dtype=np.float64).reshape(1, 1, -1)

    diff = yt[:, :, None] - yq
    per_target_tau = np.mean(
        np.maximum(taus_arr * diff, (taus_arr - 1.0) * diff),
        axis=0,
    )  # shape (n_targets, n_taus)

    per_target: dict[str, dict] = {}
    for i, name in enumerate(names):
        per_tau = {tau: float(per_target_tau[i, j]) for j, tau in enumerate(taus_list)}
        per_target[name] = {
            "mean_pinball": float(per_target_tau[i].mean()),
            "per_tau": per_tau,
        }

    return {
        "taus": taus_list,
        "per_target": per_target,
        "aggregate": {"mean_pinball": float(per_target_tau.mean(axis=1).mean())},
    }


def pit_coverage(
    y_true: np.ndarray | pd.DataFrame,
    y_quantiles: np.ndarray,
    levels: Sequence[float],
    target_names: Sequence[str] | None = None,
) -> dict:
    """Empirical cumulative coverage ``P(y <= q_τ)`` versus nominal ``τ``.

    For a calibrated model, empirical coverage at each nominal level ``τ``
    should equal ``τ``. The returned ``per_target`` and ``aggregate`` values
    pair directly with a nominal-vs-empirical reliability diagram.
    """
    yt = _as_2d(y_true)
    yq = np.asarray(y_quantiles, dtype=np.float64)
    if yq.ndim == 2:
        yq = yq.reshape(yq.shape[0], 1, yq.shape[1])
    if yq.shape[0] != yt.shape[0] or yq.shape[1] != yt.shape[1]:
        raise ValueError(
            f"y_quantiles shape {yq.shape} incompatible with y_true shape {yt.shape}",
        )
    if yq.shape[2] != len(levels):
        raise ValueError(
            f"y_quantiles has {yq.shape[2]} quantiles but {len(levels)} levels",
        )

    names = _resolve_target_names(y_true, yt.shape[1], target_names)
    levels_list = [float(lv) for lv in levels]

    per_target: dict[str, dict[float, float]] = {}
    emp_per_level: dict[float, list[float]] = {lv: [] for lv in levels_list}
    for i, name in enumerate(names):
        level_cov: dict[float, float] = {}
        for j, lv in enumerate(levels_list):
            cov = float(np.mean(yt[:, i] <= yq[:, i, j]))
            level_cov[lv] = cov
            emp_per_level[lv].append(cov)
        per_target[name] = level_cov

    aggregate: dict[float, float] = {
        lv: float(np.mean(emp_per_level[lv])) for lv in levels_list
    }
    return {
        "nominal": levels_list,
        "per_target": per_target,
        "aggregate": aggregate,
    }


def pa_checkpoint_rows(
    snapshots: pd.DataFrame,
    thresholds: Sequence[int] = DEFAULT_PA_CHECKPOINTS,
    group_keys: Sequence[str] = ("mlbam_id", "season"),
    pa_col: str = "pa_ytd",
) -> dict[int, pd.DataFrame]:
    """Select per-player rows at PA checkpoints.

    For each threshold ``T``, return the **first** weekly row per
    ``group_keys`` where ``pa_col >= T``. Rows are sorted by
    ``(iso_year, iso_week)`` within each group before the first-crossing
    selection.

    Parameters
    ----------
    snapshots:
        Weekly snapshot DataFrame (e.g. from ``weekly_snapshots_{year}.parquet``).
    thresholds:
        PA cumulative thresholds to evaluate at, in increasing order.
    group_keys:
        Columns identifying a single player-season.
    pa_col:
        Column holding the cumulative PA count (default ``pa_ytd``).

    Returns
    -------
    dict[int, pd.DataFrame]
        ``{threshold: subset DataFrame}``. Each subset has at most one row
        per player-season (players who never reach the threshold are absent).

    Notes
    -----
    The plan calls for interpolation to "week endpoints that cross those
    thresholds", which is exactly first-crossing selection: the chosen week
    is the earliest in which ``pa_ytd >= threshold``. Per-week snapshots
    are evaluated at week boundaries, not mid-week.
    """
    if pa_col not in snapshots.columns:
        raise KeyError(f"{pa_col!r} missing from snapshots")

    sort_cols = [*group_keys, "iso_year", "iso_week"]
    missing = [c for c in sort_cols if c not in snapshots.columns]
    if missing:
        raise KeyError(f"snapshots is missing sort columns: {missing}")

    sorted_df = snapshots.sort_values(sort_cols).reset_index(drop=True)

    result: dict[int, pd.DataFrame] = {}
    for t in thresholds:
        mask = sorted_df[pa_col] >= t
        # First-crossing row per group: keep=first within the masked subset.
        hit = sorted_df.loc[mask].drop_duplicates(
            subset=list(group_keys), keep="first",
        ).reset_index(drop=True)
        result[int(t)] = hit
    return result



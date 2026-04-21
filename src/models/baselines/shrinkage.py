"""Closed-form Bayesian shrinkage ROS baseline.

For each of the six ROS rate targets, combines the preseason MTL prediction
(as a Beta prior) with observed year-to-date counts (likelihood) to produce
a posterior-mean estimate of the true rate. The prior weight ``tau0`` is a
per-stat pseudocount that controls how much observed PA is needed to move
the posterior away from the preseason prior; it can be either set to
stabilisation-based defaults or fit on held-out data via ``fit_tau_per_stat``.

Formula
-------
``posterior_mean = (tau0 * preseason_rate + obs_successes) / (tau0 + obs_trials)``

Successes/trials per stat
-------------------------
* **OBP** — successes = H + BB + HBP, trials = AB + BB + HBP + SF
* **SLG** — successes = 1B + 2·2B + 3·3B + 4·HR (total bases), trials = AB
* **HR/R/RBI/SB per PA** — successes = stat count, trials = PA

All counts pulled from ``*_ytd`` columns in the weekly-snapshot schema.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from src.eval.ros_metrics import ROS_RATE_TARGETS, ROS_TARGET_STATS

# Per-stat pseudocount defaults (units = "equivalent observed trials").
# Roughly 1.5x the FanGraphs stabilisation point for each stat, which is
# where half the variance in the observed rate comes from noise.
DEFAULT_TAU0: dict[str, float] = {
    "obp": 300.0,
    "slg": 400.0,
    "hr": 150.0,
    "r": 300.0,
    "rbi": 300.0,
    "sb": 100.0,
}

# Sensible search bounds when fitting tau0. Too small and the posterior
# degenerates to the ytd rate; too large and it freezes to preseason.
_TAU_SEARCH_BOUNDS: tuple[float, float] = (1.0, 5000.0)


def shrinkage_posterior_mean(
    preseason_rate: pd.Series,
    successes: pd.Series,
    trials: pd.Series,
    tau0: float,
) -> pd.Series:
    """Beta-Binomial posterior mean with pseudocount ``tau0``.

    Equivalent to a trial-count-weighted blend between the preseason prior
    and the observed ytd rate. When ``trials == 0`` the result is the prior;
    as ``trials -> inf`` the result approaches the observed rate.
    """
    pre = preseason_rate.astype(float)
    succ = successes.astype(float)
    tri = trials.astype(float)
    denom = tau0 + tri
    return (tau0 * pre + succ) / denom


def ytd_successes_trials(rows: pd.DataFrame, stat: str) -> tuple[pd.Series, pd.Series]:
    """Return (successes, trials) Series for ``stat`` using ytd snapshot columns.

    Any NaN count is treated as 0 — a player-week with missing subtotals is
    treated as contributing no evidence for that stat, not as missing data.
    """
    def col(name: str) -> pd.Series:
        if name not in rows.columns:
            return pd.Series(0.0, index=rows.index)
        return rows[name].fillna(0.0).astype(float)

    if stat == "obp":
        successes = col("h_ytd") + col("bb_ytd") + col("hbp_ytd")
        trials = col("ab_ytd") + col("bb_ytd") + col("hbp_ytd") + col("sf_ytd")
    elif stat == "slg":
        successes = (
            col("singles_ytd")
            + 2.0 * col("doubles_ytd")
            + 3.0 * col("triples_ytd")
            + 4.0 * col("hr_ytd")
        )
        trials = col("ab_ytd")
    elif stat in ("hr", "r", "rbi", "sb"):
        successes = col(f"{stat}_ytd")
        trials = col("pa_ytd")
    else:
        raise KeyError(f"Unsupported stat {stat!r}; expected one of {ROS_TARGET_STATS}")
    return successes, trials


class ShrinkageBaseline:
    """Closed-form Bayesian shrinkage predictor for the six ROS rate targets."""

    def __init__(self, tau_per_stat: dict[str, float] | None = None) -> None:
        self.tau_per_stat: dict[str, float] = {
            **DEFAULT_TAU0,
            **(tau_per_stat or {}),
        }

    def predict(
        self,
        rows: pd.DataFrame,
        preseason: pd.DataFrame,
        id_col: str = "mlbam_id",
    ) -> pd.DataFrame | None:
        """Predict ROS rates for checkpoint ``rows`` using ``preseason`` priors.

        Returns a DataFrame with columns ordered to match ``ROS_RATE_TARGETS``.
        Unmatched players (no preseason row for their ``id_col``) yield all-NaN
        rows. Returns ``None`` when:

        * the join key is missing on either side,
        * any required ``target_*`` column is absent from ``preseason``, or
        * the preseason and checkpoint rows have zero ID overlap.
        """
        if id_col not in rows.columns or id_col not in preseason.columns:
            return None

        pre_target_cols = [f"target_{s}" for s in ROS_TARGET_STATS]
        missing_cols = [c for c in pre_target_cols if c not in preseason.columns]
        if missing_cols:
            return None

        pre_aligned = (
            preseason.drop_duplicates(subset=[id_col])
            .set_index(id_col)[pre_target_cols]
            .reindex(rows[id_col].values)
            .reset_index(drop=True)
        )
        if pre_aligned.isna().all(axis=None):
            return None

        out = pd.DataFrame(index=rows.index, columns=list(ROS_RATE_TARGETS), dtype=float)
        for stat, ros_col, pre_col in zip(
            ROS_TARGET_STATS,
            ROS_RATE_TARGETS,
            pre_target_cols,
        ):
            succ, trials = ytd_successes_trials(rows, stat)
            posterior = shrinkage_posterior_mean(
                preseason_rate=pre_aligned[pre_col].reset_index(drop=True),
                successes=succ.reset_index(drop=True),
                trials=trials.reset_index(drop=True),
                tau0=self.tau_per_stat[stat],
            )
            out[ros_col] = posterior.values
        return out


def fit_tau_per_stat(
    training_rows: pd.DataFrame,
    preseason: pd.DataFrame,
    stats: Iterable[str] | None = None,
    id_col: str = "mlbam_id",
    bounds: tuple[float, float] = _TAU_SEARCH_BOUNDS,
) -> dict[str, float]:
    """Fit ``tau0`` per stat by minimizing RMSE of posterior mean vs actual ROS rate.

    One scalar bounded minimization per stat via scipy's Brent method.
    Rows missing either the preseason prior or the observed ROS target for a
    stat are skipped for that stat; a stat with no usable rows falls back to
    its ``DEFAULT_TAU0``.
    """
    stats = list(stats) if stats is not None else list(ROS_TARGET_STATS)

    pre_indexed = preseason.drop_duplicates(subset=[id_col]).set_index(id_col)
    joined = training_rows.merge(
        pre_indexed[[f"target_{s}" for s in stats]],
        left_on=id_col,
        right_index=True,
        how="inner",
    )

    fitted: dict[str, float] = {}
    for stat in stats:
        ros_col = ROS_RATE_TARGETS[ROS_TARGET_STATS.index(stat)]
        pre_col = f"target_{stat}"
        if ros_col not in joined.columns:
            fitted[stat] = DEFAULT_TAU0[stat]
            continue
        mask = joined[[pre_col, ros_col]].notna().all(axis=1)
        if not mask.any():
            fitted[stat] = DEFAULT_TAU0[stat]
            continue
        subset = joined.loc[mask].reset_index(drop=True)
        pre_rate = subset[pre_col].astype(float)
        y_true = subset[ros_col].astype(float)
        succ, trials = ytd_successes_trials(subset, stat)

        def loss(tau: float, pre_rate=pre_rate, succ=succ, trials=trials, y_true=y_true) -> float:
            preds = shrinkage_posterior_mean(pre_rate, succ, trials, tau0=float(tau))
            return float(np.sqrt(np.mean((preds.values - y_true.values) ** 2)))

        result = minimize_scalar(loss, bounds=bounds, method="bounded")
        fitted[stat] = float(result.x)
    return fitted

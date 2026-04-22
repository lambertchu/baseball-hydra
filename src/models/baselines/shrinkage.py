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

In addition to the posterior-mean point estimate, this module exposes a
``shrinkage_posterior_quantiles`` helper that returns Beta-CDF quantiles of
the posterior distribution for uncertainty-aware reporting (pinball loss,
PIT coverage).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import beta as _beta

from src.eval.metrics import rmse
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
    return (tau0 * pre + succ) / (tau0 + tri)


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


def _align_preseason(
    rows: pd.DataFrame,
    preseason: pd.DataFrame,
    id_col: str,
    season_col: str = "season",
) -> pd.DataFrame | None:
    """Broadcast preseason rate predictions onto the checkpoint row order.

    The join is keyed on ``(id_col, season_col)`` when both frames carry
    ``season`` so multi-season rows don't collapse onto the same player's
    first-season prior (which would attach the wrong year's prior to a
    checkpoint row). When either side lacks ``season`` we fall back to the
    id-only join, which is safe for single-year calls.

    Returns a DataFrame with ``ROS_RATE_TARGETS`` columns, or ``None`` when
    the join is unusable (missing key, missing targets, zero overlap).
    """
    if id_col not in rows.columns or id_col not in preseason.columns:
        return None
    pre_target_cols = [f"target_{s}" for s in ROS_TARGET_STATS]
    if any(c not in preseason.columns for c in pre_target_cols):
        return None

    use_season = season_col in rows.columns and season_col in preseason.columns
    if use_season:
        key_cols = [id_col, season_col]
        pre_dedup = preseason.drop_duplicates(subset=key_cols).set_index(key_cols)[
            pre_target_cols
        ]
        # Build a MultiIndex from the rows' (id, season) pairs in the
        # checkpoint row order; reindex preserves row alignment and produces
        # NaN for unmatched (player, season) combinations.
        keys = pd.MultiIndex.from_arrays(
            [rows[id_col].values, rows[season_col].values],
            names=key_cols,
        )
        aligned = pre_dedup.reindex(keys).reset_index(drop=True)
    else:
        aligned = (
            preseason.drop_duplicates(subset=[id_col])
            .set_index(id_col)[pre_target_cols]
            .reindex(rows[id_col].values)
            .reset_index(drop=True)
        )
    aligned.columns = list(ROS_RATE_TARGETS)
    if aligned.isna().all(axis=None):
        return None
    return aligned


def predict_shrinkage(
    rows: pd.DataFrame,
    preseason: pd.DataFrame | None = None,
    tau_per_stat: dict[str, float] | None = None,
    id_col: str = "mlbam_id",
    preseason_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Predict ROS rates for checkpoint ``rows`` via Bayesian shrinkage.

    Pass an already-aligned ``preseason_matrix`` (columns named after
    ``ROS_RATE_TARGETS``) to reuse work a caller has done — for example
    the benchmark harness shares the same aligned frame between the frozen,
    marcel-blend, and shrinkage baselines. Otherwise supply the raw
    ``preseason`` DataFrame and alignment happens internally.

    Returns ``None`` when the preseason cache is unusable (missing join key,
    missing ``target_*`` columns, zero ID overlap).
    """
    if preseason_matrix is None:
        if preseason is None:
            raise ValueError("predict_shrinkage requires preseason or preseason_matrix")
        preseason_matrix = _align_preseason(rows, preseason, id_col=id_col)
        if preseason_matrix is None:
            return None

    tau = {**DEFAULT_TAU0, **(tau_per_stat or {})}
    out = pd.DataFrame(index=rows.index, columns=list(ROS_RATE_TARGETS), dtype=float)
    for stat, ros_col in zip(ROS_TARGET_STATS, ROS_RATE_TARGETS):
        succ, trials = ytd_successes_trials(rows, stat)
        posterior = shrinkage_posterior_mean(
            preseason_rate=preseason_matrix[ros_col].reset_index(drop=True),
            successes=succ.reset_index(drop=True),
            trials=trials.reset_index(drop=True),
            tau0=tau[stat],
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

    One scalar bounded minimization per stat via scipy's Brent method. A stat
    whose ``target_{stat}`` column is missing from ``preseason``, or that
    has no usable (non-NaN) rows after the join, falls back to its
    ``DEFAULT_TAU0``.

    When both ``training_rows`` and ``preseason`` include a ``season``
    column, the join is keyed on ``(id_col, season)`` so multi-year fits
    align each observation with its correct-year prior rather than
    collapsing a player's seasons together.
    """
    stats = list(stats) if stats is not None else list(ROS_TARGET_STATS)
    fitted: dict[str, float] = {s: DEFAULT_TAU0[s] for s in stats}

    fittable = [s for s in stats if f"target_{s}" in preseason.columns]
    if not fittable:
        return fitted

    join_cols = [id_col]
    if "season" in training_rows.columns and "season" in preseason.columns:
        join_cols.append("season")

    pre_target_cols = [f"target_{s}" for s in fittable]
    pre_subset = preseason[join_cols + pre_target_cols].drop_duplicates(
        subset=join_cols
    )
    joined = training_rows.merge(pre_subset, on=join_cols, how="inner")

    for stat in fittable:
        ros_col = ROS_RATE_TARGETS[ROS_TARGET_STATS.index(stat)]
        pre_col = f"target_{stat}"
        if ros_col not in joined.columns:
            continue
        mask = joined[[pre_col, ros_col]].notna().all(axis=1)
        if not mask.any():
            continue
        subset = joined.loc[mask]
        # Pre-cast to numpy once so the bounded Brent search (15-30
        # evaluations) doesn't re-float the same Series on every call.
        pre_arr = subset[pre_col].to_numpy(dtype=float)
        y_arr = subset[ros_col].to_numpy(dtype=float)
        succ, trials = ytd_successes_trials(subset, stat)
        succ_arr = succ.to_numpy(dtype=float)
        trials_arr = trials.to_numpy(dtype=float)

        def loss(tau: float, p=pre_arr, s=succ_arr, t=trials_arr, y=y_arr) -> float:
            preds = (tau * p + s) / (tau + t)
            return rmse(y, preds)

        result = minimize_scalar(loss, bounds=bounds, method="bounded")
        fitted[stat] = float(result.x)
    return fitted


# ---------------------------------------------------------------------------
# Beta posterior quantiles
# ---------------------------------------------------------------------------

# Minimum floor for ``alpha`` / ``beta`` Beta parameters. ``scipy.stats.beta.ppf``
# returns NaN for non-positive parameters; clipping keeps downstream metrics
# defined even in degenerate edge cases (e.g. negative preseason_rate from a
# noisy MTL run, or tau0=0).
_POSTERIOR_PARAM_FLOOR: float = 1e-3

DEFAULT_QUANTILE_TAUS: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)


def _quantile_column_name(tau: float) -> str:
    """Stable column name for a tau quantile (e.g. 0.05 → ``q0_05``).

    Two-decimal formatting captures the conventional 5/25/50/75/95 grid
    without introducing float-string noise.
    """
    return f"q{tau:.2f}".replace(".", "_")


def shrinkage_posterior_quantiles(
    preseason_rate: pd.Series | np.ndarray,
    successes: pd.Series | np.ndarray,
    trials: pd.Series | np.ndarray,
    tau0: float,
    taus: Sequence[float] = DEFAULT_QUANTILE_TAUS,
) -> pd.DataFrame:
    """Beta posterior quantiles for each row.

    Builds the Beta(α, β) posterior implied by the pseudocount-weighted
    prior and observed counts, then returns ``scipy.stats.beta.ppf(tau, α, β)``
    at each requested ``tau`` level.

    Parameters
    ----------
    preseason_rate:
        Per-row preseason rate (prior mean). Will be cast to float.
    successes, trials:
        Observed counts from the weekly-snapshot schema.
    tau0:
        Per-stat pseudocount controlling prior strength.
    taus:
        Quantile levels to evaluate. Output columns follow
        :func:`_quantile_column_name` — e.g. ``q0_05`` for ``tau=0.05``.

    Returns
    -------
    pd.DataFrame
        One column per tau, same row index as ``preseason_rate`` (or
        :func:`range(len)` for ndarray inputs). All values live in ``[0, 1]``
        because the Beta distribution is supported on the unit interval —
        which is the right support for rate targets including per-PA rates.
    """
    if isinstance(preseason_rate, pd.Series):
        idx = preseason_rate.index
    elif isinstance(successes, pd.Series):
        idx = successes.index
    elif isinstance(trials, pd.Series):
        idx = trials.index
    else:
        idx = pd.RangeIndex(len(np.asarray(preseason_rate)))

    pre_arr = np.asarray(preseason_rate, dtype=np.float64)
    succ_arr = np.asarray(successes, dtype=np.float64)
    trials_arr = np.asarray(trials, dtype=np.float64)

    # Prior parameters: alpha0 = tau0 * p, beta0 = tau0 * (1 - p).
    # Posterior: alpha = alpha0 + successes, beta = beta0 + (trials - successes).
    alpha = tau0 * pre_arr + succ_arr
    beta = tau0 * (1.0 - pre_arr) + (trials_arr - succ_arr)

    # Clip to a positive floor so scipy's ppf stays well-defined.
    alpha = np.clip(alpha, _POSTERIOR_PARAM_FLOOR, None)
    beta = np.clip(beta, _POSTERIOR_PARAM_FLOOR, None)

    cols = {}
    for tau in taus:
        cols[_quantile_column_name(float(tau))] = _beta.ppf(float(tau), alpha, beta)
    return pd.DataFrame(cols, index=idx)


def predict_shrinkage_quantiles(
    rows: pd.DataFrame,
    preseason: pd.DataFrame | None = None,
    tau_per_stat: dict[str, float] | None = None,
    id_col: str = "mlbam_id",
    taus: Sequence[float] = DEFAULT_QUANTILE_TAUS,
    preseason_matrix: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame] | None:
    """Predict ROS rate quantiles for checkpoint ``rows`` via Beta posterior.

    Returns a dict keyed by ROS rate target name (e.g. ``"ros_obp"``) with a
    per-target DataFrame of quantile predictions. Mirrors the alignment
    semantics of :func:`predict_shrinkage` — passing a ``preseason_matrix``
    lets the caller reuse an alignment it has already done, otherwise the
    raw ``preseason`` frame is aligned internally.

    Returns
    -------
    dict[str, pd.DataFrame] | None
        ``{ros_target: DataFrame[quantile_columns]}`` with one row per
        entry of ``rows`` and one column per tau level. ``None`` when the
        preseason cache is unusable (missing join key, missing target
        columns, zero ID overlap).
    """
    if preseason_matrix is None:
        if preseason is None:
            raise ValueError(
                "predict_shrinkage_quantiles requires preseason or preseason_matrix"
            )
        preseason_matrix = _align_preseason(rows, preseason, id_col=id_col)
        if preseason_matrix is None:
            return None

    tau = {**DEFAULT_TAU0, **(tau_per_stat or {})}
    out: dict[str, pd.DataFrame] = {}
    for stat, ros_col in zip(ROS_TARGET_STATS, ROS_RATE_TARGETS):
        succ, trials = ytd_successes_trials(rows, stat)
        q_df = shrinkage_posterior_quantiles(
            preseason_rate=preseason_matrix[ros_col].reset_index(drop=True),
            successes=succ.reset_index(drop=True),
            trials=trials.reset_index(drop=True),
            tau0=tau[stat],
            taus=taus,
        )
        q_df.index = rows.index
        out[ros_col] = q_df
    return out

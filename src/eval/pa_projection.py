"""PA projection and rate-to-count conversion utilities.

Used by both the projection script (``generate_projections.py``) and the
benchmark script (``benchmark_vs_public.py``) to convert per-PA rate
predictions to counting stats via projected PA.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src.features.registry import COUNT_STATS

# Default full-season games
_FULL_SEASON_GAMES = 162
_MAX_PA = 750

# Season files only; data/raw also holds batting_week_YYYY.parquet.
_HISTORY_FILE = re.compile(r"batting_\d{4}\.parquet")
_HISTORY_COLS = ["idfg", "season", "pa", "g", "age", "war", "wrc_plus"]
PA_FEATURES = [
    "pa_y1",
    "pa_y2",
    "pa_y3",
    "g_y1",
    "war_y1",
    "wrc_plus_y1",
    "age",
    "age_sq",
]


def project_pa(
    df: pd.DataFrame,
    season_games: dict[int, int] | None = None,
) -> pd.Series:
    """Project next-season PA using the Marcel formula.

    ``PA_proj = 0.5 * PA_y1 + 0.1 * PA_y2 + 200``

    For shortened seasons (e.g. 2020), the PA is scaled to 162-game pace
    before applying the formula so that a 250-PA 60-game season is treated
    as ~675-pace, producing a reasonable full-season projection.

    Parameters
    ----------
    df:
        DataFrame with ``pa`` (current-year PA), ``season``, and optionally
        ``prev_year_pa`` for Y-2.
    season_games:
        Mapping of season year → actual games played for shortened seasons.
        Years not in this dict are assumed to be 162-game seasons.

    Returns
    -------
    pd.Series
        Projected PA for each player.
    """
    if season_games is None:
        season_games = {}

    pa_y1 = df["pa"].astype(float).copy()

    # Scale shortened-season PA to 162-game pace for the Marcel formula
    if season_games and "season" in df.columns:
        for year, games in season_games.items():
            mask = df["season"] == year
            if mask.any() and games < _FULL_SEASON_GAMES:
                pa_y1.loc[mask] = pa_y1.loc[mask] * (_FULL_SEASON_GAMES / games)

    # Y-2 PA: use prev_year_pa if available (already computed by temporal features)
    if "prev_year_pa" in df.columns:
        pa_y2 = df["prev_year_pa"].fillna(0).astype(float)
    else:
        pa_y2 = pd.Series(0.0, index=df.index)

    # Marcel formula: PA_proj = 0.5 * PA_y1 + 0.1 * PA_y2 + 200
    projected = 0.5 * pa_y1 + 0.1 * pa_y2 + 200.0

    # Clamp to reasonable range (200-750 PA)
    projected = projected.clip(lower=200, upper=750)

    return projected


def load_batting_history(raw_dir: str | Path) -> pd.DataFrame:
    """Stack every ``batting_YYYY.parquet`` season file (all hitters, qual=0)."""
    frames = [
        pd.read_parquet(path, columns=_HISTORY_COLS)
        for path in sorted(Path(raw_dir).glob("batting_*.parquet"))
        if _HISTORY_FILE.fullmatch(path.name)
    ]
    history = pd.concat(frames, ignore_index=True)
    history["idfg"] = history["idfg"].astype(int)
    history["season"] = history["season"].astype(int)
    return history


def pa_features(
    keys: pd.DataFrame,
    history: pd.DataFrame,
    season_games: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build PA-model features for ``(idfg, season)`` keys from batting history.

    PA, games, and WAR are scaled to 162-game pace. A prior season inside the
    history range with no row means 0 PA; one before the range is NaN.
    """
    season_games = season_games or {}
    first_season = int(history["season"].min())
    hist = history.copy()
    games = hist["season"].map(season_games).fillna(_FULL_SEASON_GAMES)
    for col in ("pa", "g", "war"):
        hist[col] = hist[col] * (_FULL_SEASON_GAMES / games)
    hist = hist.set_index(["idfg", "season"])

    idfg = keys["idfg"].astype(int).to_numpy()
    season = keys["season"].astype(int).to_numpy()

    def lookup(col: str, lag: int, fill_absent: bool) -> np.ndarray:
        years = season - lag
        idx = pd.MultiIndex.from_arrays([idfg, years])
        vals = hist[col].reindex(idx).to_numpy(dtype=float)
        if fill_absent:
            vals = np.where(np.isnan(vals) & (years >= first_season), 0.0, vals)
        return vals

    feats = pd.DataFrame(
        {
            "pa_y1": lookup("pa", 0, True),
            "pa_y2": lookup("pa", 1, True),
            "pa_y3": lookup("pa", 2, True),
            "g_y1": lookup("g", 0, True),
            "war_y1": lookup("war", 0, False),
            "wrc_plus_y1": lookup("wrc_plus", 0, False),
            "age": lookup("age", 0, False),
        },
        index=keys.index,
    )
    feats["age_sq"] = feats["age"] ** 2
    return feats[PA_FEATURES]


def fit_pa_model(
    history: pd.DataFrame,
    max_label_year: int,
    season_games: dict[int, int] | None,
    cfg: dict,
    seed: int = 42,
) -> LinearRegression | HistGradientBoostingRegressor:
    """Fit next-season PA on history rows whose label year is ``<= max_label_year``.

    The label is actual PA in Y+1, or 0 when the player has no row that year.
    Shortened label seasons are skipped.
    """
    season_games = season_games or {}
    short = [y for y, g in season_games.items() if g < _FULL_SEASON_GAMES]
    label_year = history["season"] + 1
    rows = history[
        (label_year <= max_label_year)
        & ~label_year.isin(short)
        & (history["season"] - 2 >= history["season"].min())
    ]
    X = pa_features(rows, history, season_games)
    keep = X["pa_y1"] >= cfg.get("min_pa_train", 200)
    rows, X = rows[keep], X[keep]

    next_pa = history.set_index(["idfg", "season"])["pa"]
    label_idx = pd.MultiIndex.from_arrays([rows["idfg"], rows["season"] + 1])
    y = next_pa.reindex(label_idx).fillna(0.0).to_numpy()

    if cfg.get("estimator", "ols") == "gbm":
        model = HistGradientBoostingRegressor(**cfg.get("gbm", {}), random_state=seed)
    else:
        model = LinearRegression()
    return model.fit(X.to_numpy(), y)


def project_next_season_pa(
    df: pd.DataFrame,
    data_config: dict,
    max_label_year: int,
) -> pd.Series:
    """Project next-season PA with the method in ``data_config["pa_model"]``.

    ``max_label_year`` is the last season usable as a training label. Pass the
    season of ``df``'s rows (the year before the projected one) to avoid leakage.
    """
    cfg = data_config.get("pa_model", {})
    season_games = data_config.get("season_games", {})
    if cfg.get("method", "marcel") == "marcel":
        return project_pa(df, season_games)

    history = load_batting_history(data_config.get("raw_data_dir", "data/raw"))
    model = fit_pa_model(
        history, max_label_year, season_games, cfg, data_config.get("seed", 42)
    )
    X = pa_features(df, history, season_games)
    projected = np.clip(model.predict(X.to_numpy()), 0, _MAX_PA)
    return pd.Series(projected, index=df.index)


def rate_to_count(
    rate_predictions: pd.DataFrame,
    projected_pa: pd.Series,
) -> pd.DataFrame:
    """Convert per-PA rate predictions to counting stats.

    Parameters
    ----------
    rate_predictions:
        Model output with columns ``target_hr``, ``target_r``, etc.
        containing per-PA rates.
    projected_pa:
        Projected PA for each player.

    Returns
    -------
    pd.DataFrame
        Predictions with count targets converted to raw counts.
        OBP and SLG are left unchanged.
    """
    result = rate_predictions.copy()
    for stat in COUNT_STATS:
        col = f"target_{stat}"
        if col in result.columns:
            result[col] = result[col] * projected_pa.values
    return result

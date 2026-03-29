"""PA projection and rate-to-count conversion utilities.

Used by both the projection script (``generate_projections.py``) and the
benchmark script (``benchmark_vs_public.py``) to convert per-PA rate
predictions to counting stats via Marcel PA projection.
"""
from __future__ import annotations

import pandas as pd

from src.features.registry import COUNT_STATS

# Default full-season games
_FULL_SEASON_GAMES = 162


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

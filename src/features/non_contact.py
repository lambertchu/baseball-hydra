"""Compute stabilisation-regressed non-contact rate features.

Plate discipline outcomes (strikeouts, walks, hit-by-pitches) stabilise at
different sample sizes. This module applies regression-to-the-mean using
stabilisation points from FanGraphs research (via pwOBA methodology):

  regressed_rate = (PA * observed + stab * lg_avg) / (PA + stab)

where ``stab`` is the number of PA at which the observed rate is 50%
signal / 50% noise. Strikeout rate stabilises quickly (60 PA), while
HBP rate requires much larger samples (300 PA).

These regressed rates are more predictive than raw observed rates,
especially for low-PA batters where sample noise dominates.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Stabilisation points: PA at which observed rate is 50% signal
STABILISATION_PA: dict[str, int] = {
    "k_rate": 60,
    "bb_rate": 120,
    "hbp_rate": 300,
    "babip": 1200,
    "iso": 160,
}

# League-average rates (approximate MLB averages)
LEAGUE_AVG: dict[str, float] = {
    "k_rate": 0.224,
    "bb_rate": 0.078,
    "hbp_rate": 0.012,
    "babip": 0.300,
    "iso": 0.150,
}

# Stabilisation for BBE-denominator stats (separate dict since denominator
# is batted ball events, not plate appearances)
STABILISATION_BBE: dict[str, int] = {
    "hr_per_bbe": 60,
}

LEAGUE_AVG_BBE: dict[str, float] = {
    "hr_per_bbe": 0.045,
}


def regress_to_mean(observed: float, pa: float, stat: str) -> float:
    """Apply stabilisation-point regression to the mean.

    Parameters
    ----------
    observed:
        Observed rate for the batter-season.
    pa:
        Plate appearances in the season.
    stat:
        Rate stat name (must be a key in ``STABILISATION_PA``).

    Returns
    -------
    float
        Regressed rate blending observed data with league average.
    """
    stab = STABILISATION_PA[stat]
    lg_avg = LEAGUE_AVG[stat]
    return (pa * observed + stab * lg_avg) / (pa + stab)


def compute_non_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute stabilisation-regressed non-contact rate features.

    Requires ``pa``, ``k_rate``, ``bb_rate``, and ``hbp_rate`` columns
    to already be present (batting features must run first).

    Parameters
    ----------
    df:
        DataFrame with batting features already computed.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``regressed_k_rate``, ``regressed_bb_rate``,
        and ``regressed_hbp_rate`` columns added.
    """
    df = df.copy()

    for stat in STABILISATION_PA:
        output_col = f"regressed_{stat}"
        if stat not in df.columns:
            logger.warning(
                "Column '%s' not found — filling regressed rate with league average",
                stat,
            )
            df[output_col] = LEAGUE_AVG[stat]
            continue

        pa = df["pa"].fillna(0).astype(float) if "pa" in df.columns else pd.Series(0.0, index=df.index)
        observed = df[stat].fillna(LEAGUE_AVG[stat]).astype(float)
        stab = STABILISATION_PA[stat]
        lg_avg = LEAGUE_AVG[stat]

        df[output_col] = (pa * observed + stab * lg_avg) / (pa + stab)

    # BBE-denominator stats (HR per batted ball event)
    for stat, stab in STABILISATION_BBE.items():
        output_col = f"regressed_{stat}"
        lg_avg = LEAGUE_AVG_BBE[stat]

        if stat == "hr_per_bbe":
            if "hr" in df.columns and "bbe_count" in df.columns:
                bbe = df["bbe_count"].fillna(0).astype(float)
                hr = df["hr"].fillna(0).astype(float)
                observed = np.where(bbe > 0, hr / bbe, lg_avg)
                df[output_col] = (bbe * observed + stab * lg_avg) / (bbe + stab)
            else:
                logger.warning("hr or bbe_count missing — filling %s with league avg", output_col)
                df[output_col] = lg_avg

    return df
